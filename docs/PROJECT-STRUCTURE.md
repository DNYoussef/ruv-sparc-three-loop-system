# ruv-sparc-three-loop-system - Project Structure Guide

**Plugin Status**: Production-ready with 131 agents + 111 skills
**Last Updated**: 2025-11-02
**Organization Status**: Fully organized (Phases 1-4 complete)

---

## Overview

This Claude Code plugin implements the SPARC methodology with Three-Loop development system, featuring:
- **131 specialized agents** organized into 24 domains
- **111 enhanced skills** at Silver+ tier (MECE template)
- **Multi-agent coordination** via claude-flow MCP
- **Automated enhancement pipeline** for skills

---

## Directory Structure

```
ruv-sparc-three-loop-system/
├── agents/                           # 131 agents organized by domain (24 directories)
│   ├── analysis/                     # Analytical & audit agents
│   ├── architecture/                 # System architecture & design
│   ├── business/                     # Business logic & domain
│   ├── consensus/                    # Consensus protocols (byzantine, raft, gossip)
│   ├── core/                         # Foundation agents (coder, reviewer, researcher)
│   ├── data/                         # Data processing & transformation
│   ├── database/                     # Database specialists
│   ├── development/                  # General development (backend, mobile, ML)
│   ├── devops/                       # DevOps & infrastructure
│   ├── documentation/                # Documentation generation
│   ├── flow-nexus/                   # Multi-model integration (Gemini, Codex)
│   ├── frontend/                     # Frontend specialists (React, Vue, UI)
│   ├── github/                       # GitHub integration
│   ├── goal/                         # Goal-oriented & planning
│   ├── hive-mind/                    # Collective intelligence coordination
│   ├── neural/                       # Neural network & ML models
│   ├── optimization/                 # Performance optimization
│   ├── reasoning/                    # Reasoning & decision-making
│   ├── research/                     # Research & analysis (Deep Research SOP)
│   ├── sparc/                        # SPARC methodology agents
│   ├── registry/                     # Agent registry management
│   │   ├── registry.json             # 131 agent definitions
│   │   ├── add-mcp-to-registry.js    # Registry management scripts
│   │   ├── update-mcp-free-only.js
│   │   ├── update-to-installed-only.js
│   │   └── MIGRATION_SUMMARY.md
│   └── README.md                     # Agent organization guide
│
├── skills/                           # 111 skills with MECE template (Silver+ tier)
│   ├── [111 skill directories]       # Each with skill.md, README.md, examples/, etc.
│   ├── _pipeline-automation/         # Enhancement automation
│   │   ├── enhance-skill.py          # Skill enhancement script
│   │   ├── cleanup-skill.py          # File reorganization
│   │   ├── audit-skill.py            # MECE structure validation
│   │   ├── batch-enhance.py          # Batch processing
│   │   ├── audits/                   # Audit report JSONs (centralized)
│   │   └── BATCH-1-COMPLETION-REPORT.md
│   └── README.md                     # Skills overview & enhancement status
│
├── docs/                             # Project documentation
│   ├── PROJECT-STRUCTURE.md          # This file
│   ├── COMPREHENSIVE-ORGANIZATION-PLAN.md  # 6-phase organization plan
│   ├── PHASE-1-DISCOVERY-REPORT.md   # Discovery findings
│   ├── PHASE-2-FILE-ORGANIZATION-PLAN.md  # File movement plan
│   ├── [Many other documentation files]
│   └── [Organized by topic]
│
├── hooks/                            # Claude Code hooks for automation
│   └── 12fa/                         # 12-factor app hooks
│       └── memory-mcp-tagging-protocol.js  # Auto-tagging for Memory MCP
│
├── scripts/                          # Utility scripts
├── tests/                            # Test files
├── CLAUDE.md                         # ⚠️ CRITICAL: Specialist agent selection guide
└── package.json                      # Node.js project configuration
```

---

## Key Files & Directories

### CLAUDE.md - Configuration & Specialist Agent Selection ⚠️
**Location**: Root directory
**Purpose**: Primary configuration for Claude Code interactions

**Critical Sections**:
1. **Specialist Agent Selection** (NEW): Table of 5 specialist types (researcher, coder, analyst, optimizer, coordinator)
2. **Agent Selection Decision Tree**: Keyword-based agent type selection
3. **Correct/Wrong Examples**: Shows proper Task tool usage
4. **MCP Tool Categories**: Memory MCP, Connascence Analyzer, Flow-Nexus integration

**Usage**: Read this first when working with agents or skills!

### agents/registry/registry.json - Agent Registry
**Location**: `agents/registry/registry.json`
**Size**: 194KB
**Content**: Metadata for all 131 agents

**Structure**:
```json
{
  "agents": [
    {
      "name": "agent-name",
      "domain": "development",
      "specialist_type": "coder",
      "capabilities": [...],
      "mcp_integration": true
    }
  ]
}
```

### skills/_pipeline-automation/ - Enhancement Pipeline
**Location**: `skills/_pipeline-automation/`
**Purpose**: Automated skill enhancement workflow

**Key Scripts**:
1. **enhance-skill.py**: Main enhancement script
   - Pre-enhancement validation
   - Agent spawn instructions
   - MECE template application

2. **cleanup-skill.py**: Post-enhancement cleanup
   - File reorganization
   - Naming fixes
   - Orphaned file handling

3. **audit-skill.py**: MECE structure validation
   - Structure checks (skill.md, README.md, examples/, etc.)
   - Content quality assessment
   - Quality tier calculation
   - GO/NO-GO decision (85% threshold)

4. **batch-enhance.py**: Batch processing
   - Process 15 skills at a time
   - Parallel agent execution
   - Batch audit reports

---

## File Naming Conventions

### Agents
- **Directory names**: Lowercase, hyphenated (e.g., `flow-nexus/`, `hive-mind/`)
- **Agent files**: Lowercase, hyphenated, .md extension (e.g., `multi-model-orchestrator.md`)
- **Registry scripts**: Kebab-case, .js extension (e.g., `add-mcp-to-registry.js`)

### Skills
- **Directory names**: Lowercase, hyphenated (e.g., `code-review-assistant/`, `agentdb-optimization/`)
- **Required files**: `skill.md`, `README.md` (exactly these names)
- **Example files**: `example-1-basic.md`, `example-2-intermediate.md`, `example-3-advanced.md`
- **Reference files**: Lowercase, hyphenated (e.g., `best-practices.md`, `api-reference.md`)
- **Graphviz files**: `workflow.dot` (standard), or descriptive names (e.g., `agent-coordination.dot`)

### Documentation
- **Markdown files**: UPPERCASE for key docs (e.g., `README.md`, `CLAUDE.md`)
- **Project docs**: UPPERCASE-WITH-HYPHENS (e.g., `PROJECT-STRUCTURE.md`, `PHASE-1-DISCOVERY-REPORT.md`)
- **Technical docs**: Lowercase, hyphenated (e.g., `memory-storage-patterns.md`)

---

## Workflow Patterns

### 1. Agent Spawning (via Claude Code Task Tool)

**CORRECT Pattern**:
```javascript
[Single Message - Parallel Specialist Execution]:
  Task("Requirements analyst", "Analyze API requirements...", "researcher")
  Task("Backend developer", "Implement REST endpoints...", "coder")
  Task("Security auditor", "Review vulnerabilities...", "analyst")
  Task("Performance engineer", "Optimize query performance...", "optimizer")
  Task("Feature coordinator", "Orchestrate 4 specialists...", "coordinator")
```

**WRONG Pattern** (DO NOT USE):
```javascript
Task("Generic agent", "...", "general-purpose")  // ❌ No such type!
```

### 2. Skill Enhancement Workflow

**Full Pipeline**:
```bash
# Step 1: Manual skill.md creation
vi skills/my-new-skill/skill.md

# Step 2: Enhance with agents
python skills/_pipeline-automation/enhance-skill.py skills/my-new-skill --tier Silver

# Step 3: Cleanup
python skills/_pipeline-automation/cleanup-skill.py skills/my-new-skill

# Step 4: Audit
python skills/_pipeline-automation/audit-skill.py skills/my-new-skill

# Step 5: Review audit report
cat skills/_pipeline-automation/audits/my-new-skill-audit.json
```

### 3. Batch Enhancement Workflow

**Process 15 skills at once**:
```bash
# Generate enhancement plans
python batch-enhance.py --batch-num 2 --plan-only

# Spawn 15 parallel agents via Claude Code Task tool
# (See Batch 1 example in BATCH-1-COMPLETION-REPORT.md)

# Cleanup all 15
for skill in $(cat batch-2-skills.txt); do
  python cleanup-skill.py ../skills/$skill
done

# Audit all 15
python batch-enhance.py --batch-num 2 --audit-only
```

---

## Quality Standards

### Agent Quality
- **Domain categorization**: Every agent in appropriate domain directory
- **Registry entry**: All agents listed in `agents/registry/registry.json`
- **Specialist type**: Every agent tagged with one of 5 types (researcher/coder/analyst/optimizer/coordinator)
- **Documentation**: .md file with purpose, capabilities, integration

### Skill Quality Tiers
| Tier | Files | Standard |
|------|-------|----------|
| **Bronze** | 3+ | skill.md + README.md + 1 example |
| **Silver** | 7+ | Bronze + 3 examples + 2 references + graphviz |
| **Gold** | 12+ | Silver + resources/ (scripts, templates) + tests/ |
| **Platinum** | 20+ | Gold + comprehensive resources + extensive tests |

**Current Status**: All 111 skills at Silver+ tier (100% complete)

### Audit Criteria
- **Structure Score** (40%): File organization, naming conventions, MECE compliance
- **Content Score** (30%): YAML frontmatter, imperative voice, concrete examples
- **Quality Tier Score** (30%): Files count, directory structure
- **Overall Threshold**: 85% for GO decision

---

## Integration Points

### MCP Servers

**Required**: claude-flow (coordination)
```bash
claude mcp add claude-flow npx claude-flow@alpha mcp start
```

**Optional**:
- ruv-swarm (enhanced coordination)
- flow-nexus (cloud features)
- memory-mcp (persistent context) - PRODUCTION READY
- connascence-analyzer (code quality) - PRODUCTION READY

### Claude Code Integration

**Primary Tools**:
- `Task` tool: Spawn specialist agents for parallel execution
- `TodoWrite`: Track multi-step workflows
- `Read`/`Write`/`Edit`: File operations
- `Bash`: Script execution, git operations
- `Skill`: Auto-invoke skills by name

### Memory MCP Integration

**Tagging Protocol** (REQUIRED for all writes):
```javascript
const { taggedMemoryStore } = require('./hooks/12fa/memory-mcp-tagging-protocol.js');

taggedMemoryStore('coder', 'Implemented authentication feature', {
  task_id: 'AUTH-123'
});
// Auto-includes: WHO (agent metadata), WHEN (timestamps), PROJECT, WHY (intent)
```

---

## Development Workflows

### Adding a New Agent

1. **Choose Domain**: Select from 24 existing domains or create new
2. **Create Agent File**: `agents/{domain}/{agent-name}.md`
3. **Update Registry**:
   ```bash
   cd agents/registry
   node add-mcp-to-registry.js --name "agent-name" --domain "domain" --type "specialist-type"
   ```
4. **Document**: Add purpose, capabilities, specialist type, integration points

### Adding a New Skill

1. **Create Directory**: `skills/{skill-name}/`
2. **Write skill.md**: YAML frontmatter + imperative voice
3. **Enhance**:
   ```bash
   python skills/_pipeline-automation/enhance-skill.py skills/{skill-name} --tier Silver
   ```
4. **Spawn Agents**: Use Claude Code Task tool (researcher/coder/analyst)
5. **Cleanup**: `python cleanup-skill.py skills/{skill-name}`
6. **Audit**: `python audit-skill.py skills/{skill-name}`
7. **Verify**: Check audit report for GO decision

### Updating Documentation

**Golden Rule**: All documentation in `/docs` directory (NOT root)

**Exception**: CLAUDE.md, README.md, package.json (remain in root)

---

## Troubleshooting

### Audit Failures

**Issue**: Skill audit shows NO-GO with valid structure
**Fix**: Check `audit-skill.py` whitelist for acceptable project directories

**Issue**: Missing skill.md causes auto-fail
**Fix**: Ensure `skill.md` (or `SKILL.md`) exists before enhancement

### Agent Spawning Issues

**Issue**: "general-purpose agent type not found"
**Fix**: Use one of 5 specialist types: researcher/coder/analyst/optimizer/coordinator

**Issue**: Agents not coordinating
**Fix**: Check claude-flow MCP server is running (`npx claude-flow@alpha status`)

### File Organization

**Issue**: Loose files in agents/ or skills/ root
**Fix**: Move to appropriate subdirectories (see PHASE-2-FILE-ORGANIZATION-PLAN.md)

---

## Maintenance

### Regular Tasks

**Weekly**:
- Verify all audit JSONs in `skills/_pipeline-automation/audits/`
- Check agent registry integrity
- Update CLAUDE.md with new patterns

**Monthly**:
- Review skill quality tiers (upgrade Bronze → Silver)
- Audit agent domain categorization
- Update specialist agent mapping table

**Per-Release**:
- Run full batch audit on all 111 skills
- Verify MCP server compatibility
- Update PROJECT-STRUCTURE.md

---

## Resources

### Documentation
- **CLAUDE.md**: Primary configuration (specialist agent selection)
- **agents/README.md**: Agent organization guide
- **skills/README.md**: Skills enhancement status
- **docs/COMPREHENSIVE-ORGANIZATION-PLAN.md**: 6-phase organization plan

### Scripts
- **skills/_pipeline-automation/**: Enhancement automation
- **agents/registry/**: Registry management scripts

### Reports
- **skills/_pipeline-automation/BATCH-1-COMPLETION-REPORT.md**: Batch 1 results
- **skills/_pipeline-automation/audits/**: Individual skill audits
- **docs/PHASE-*-*.md**: Phase reports from organization effort

---

## Change Log

### 2025-11-02: Major Organization & Enhancement
- **Phase 1-5**: Comprehensive folder organization
- **Agents**: Moved 28 loose files to proper domains
- **Skills**: Moved 10 audit JSONs to centralized location
- **CLAUDE.md**: Added specialist agent selection guide (CRITICAL)
- **Documentation**: Created agents/README.md, skills/README.md, PROJECT-STRUCTURE.md
- **Scripts**: Updated audit-skill.py with directory whitelist
- **Skills Enhancement**: Discovered all 111 skills already enhanced (Phase 6 complete)

---

**Plugin Status**: ✅ PRODUCTION READY
- 131 agents (24 domains)
- 111 skills (100% Silver+ tier)
- Full MECE compliance
- Specialist agent types enforced
- Comprehensive documentation
