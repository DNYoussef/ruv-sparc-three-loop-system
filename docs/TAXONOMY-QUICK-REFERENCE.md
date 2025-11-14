# Taxonomy Quick Reference Guide

**Date**: 2025-11-09
**Purpose**: Fast lookup for new directory structure
**Full Plan**: [TAXONOMY-REORGANIZATION-PLAN.md](TAXONOMY-REORGANIZATION-PLAN.md)

---

## 🗂️ Directory Structure at a Glance

```
PROJECT ROOT
│
├── .claude/                          # Claude Code configuration & runtime
│   ├── active/                       # Current working state (NEVER auto-archived)
│   │   ├── agents/                   # Agent definitions (131 total)
│   │   │   ├── core/                 # 8 foundational agents
│   │   │   ├── specialized/          # 45 domain specialists
│   │   │   ├── research/             # 4 Deep Research SOP agents
│   │   │   └── meta/                 # 15 orchestration agents
│   │   ├── skills/                   # Skill definitions (71 total)
│   │   │   ├── development/          # 18 development skills
│   │   │   ├── quality/              # 12 quality/testing skills
│   │   │   ├── research/             # 9 research skills
│   │   │   ├── infrastructure/       # 8 DevOps skills
│   │   │   └── meta/                 # 7 meta-skills
│   │   ├── commands/                 # Slash command definitions
│   │   ├── config/                   # Active configuration files
│   │   └── artifacts/                # Working artifacts (research, analysis)
│   │
│   ├── archive/                      # Historical preservation (MANUAL archival)
│   │   ├── 2024/                     # Year-based organization
│   │   └── 2025/Q1, Q2, Q3, Q4       # Quarter-based organization
│   │
│   └── runtime/                      # Operational data (AUTO-cleanup)
│       ├── logs/                     # System logs (retention policies)
│       ├── history/                  # Execution history
│       ├── cache/                    # Temporary cache (24h TTL)
│       └── todos/                    # Active TODO tracking
│
└── docs/                             # Documentation & knowledge
    ├── active/                       # Current project documentation
    │   ├── projects/                 # Active projects (in progress)
    │   ├── workflows/                # Active workflow documentation
    │   └── integration-plans/        # Active integration plans
    │
    ├── archive/                      # Completed documentation
    │   ├── projects/                 # Completed projects (by year/quarter)
    │   ├── research/                 # Completed research
    │   └── experiments/              # Experimental results
    │
    └── reference/                    # Timeless knowledge (NEVER archived)
        ├── agent-library/            # Agent reference docs
        ├── skill-library/            # Skill reference docs
        ├── architecture/             # Architecture patterns
        ├── methodology/              # Development methodologies
        └── tools/                    # Tool documentation
```

---

## 📋 Quick Decision Tree

### "Where does this file go?"

```
START
  │
  ├─ Is it runtime/operational data?
  │  └─ YES → .claude/runtime/
  │     ├─ Logs? → .claude/runtime/logs/
  │     ├─ History? → .claude/runtime/history/
  │     ├─ Cache? → .claude/runtime/cache/
  │     └─ TODOs? → .claude/runtime/todos/
  │
  ├─ Is it an agent definition?
  │  └─ YES → .claude/active/agents/
  │     ├─ Core agent? → .claude/active/agents/core/
  │     ├─ Specialized? → .claude/active/agents/specialized/
  │     ├─ Research? → .claude/active/agents/research/
  │     └─ Meta? → .claude/active/agents/meta/
  │
  ├─ Is it a skill definition?
  │  └─ YES → .claude/active/skills/
  │     ├─ Development? → .claude/active/skills/development/
  │     ├─ Quality? → .claude/active/skills/quality/
  │     ├─ Research? → .claude/active/skills/research/
  │     ├─ Infrastructure? → .claude/active/skills/infrastructure/
  │     └─ Meta? → .claude/active/skills/meta/
  │
  ├─ Is it configuration?
  │  └─ YES → .claude/active/config/
  │
  ├─ Is it project documentation?
  │  └─ YES → docs/
  │     ├─ Active project? → docs/active/projects/
  │     ├─ Completed project? → docs/archive/projects/YYYY/QN/
  │     └─ Workflow? → docs/active/workflows/
  │
  ├─ Is it timeless reference material?
  │  └─ YES → docs/reference/
  │     ├─ Agent docs? → docs/reference/agent-library/
  │     ├─ Skill docs? → docs/reference/skill-library/
  │     ├─ Architecture? → docs/reference/architecture/
  │     ├─ Methodology? → docs/reference/methodology/
  │     └─ Tools? → docs/reference/tools/
  │
  └─ Is it experimental/unstable?
     └─ YES → .claude/active/experimental/
```

---

## 🕐 Retention Policies (Auto-Cleanup)

| Category | Retention | Auto-Action | Frequency |
|----------|-----------|-------------|-----------|
| **Debug logs** | 7 days | Delete | Daily 2AM |
| **Error logs** | 30 days | Compress & archive | Weekly |
| **Audit logs** | 90 days | Compress & archive | Monthly |
| **Shell snapshots** | 7 days | Delete | Daily 2AM |
| **File history** | 30 days | Compress & archive | Weekly |
| **Cache** | 24 hours | Delete | Hourly |
| **Completed TODOs** | 30 days | Archive to JSON | Weekly |

**Manual Review Required**:
- Agent definitions (90+ days unused)
- Skill definitions (180+ days unused)
- Active projects (no activity 180+ days)

---

## 📑 Index Files Locations

| Index Type | Location | Purpose |
|------------|----------|---------|
| **Master Index** | `docs/INDEX.md` | Top-level navigation |
| **Agent Registry** | `.claude/active/agents/INDEX.md` | Complete agent directory |
| **Skills Catalog** | `.claude/active/skills/SKILLS-CATALOG.md` | Complete skills directory |
| **Active Projects** | `docs/active/projects/INDEX.md` | Current projects tracking |
| **Archive Index** | `docs/archive/INDEX.md` | Historical documentation |
| **Reference Index** | `docs/reference/INDEX.md` | Timeless knowledge |
| **Cross-Reference** | `docs/reference/CROSS-REFERENCE.md` | Agent↔Skill↔Command mapping |

---

## 🏷️ File Naming Conventions

### Active Content (Semantic Naming)
```
{category}-{function}-{variant}.{ext}

Examples:
- agent-core-coder-enhanced.md
- skill-testing-tdd-london-swarm.md
- workflow-development-feature-complete.md
- command-github-pr-review.md
```

### Archived Content (Date-Based Naming)
```
YYYY-MM-DD-{category}-{descriptive-name}.{ext}

Examples:
- 2025-01-15-agent-backend-api-v1.md
- 2025-03-22-skill-deprecated-old-tdd.md
- 2025-06-10-project-12fa-delivery.md
```

### Index Files (Standard Names)
```
INDEX.md               # Primary navigation
CATALOG.md             # Comprehensive metadata
SUMMARY.md             # Executive summary
QUICK-REFERENCE.md     # Quick lookup guide
README.md              # Directory overview
```

---

## 🏗️ Archival Workflow

### When to Archive

**Agents**:
- ✅ Superseded by better version
- ✅ Unused for 90+ days (check logs)
- ✅ Marked "experimental" and concluded
- ✅ Major version upgrade (archive old)

**Skills**:
- ✅ Superseded by better implementation
- ✅ Unused for 180+ days
- ✅ Marked "deprecated"
- ✅ Methodology change

**Projects**:
- ✅ Status = "completed"
- ✅ Status = "cancelled/abandoned"
- ✅ On hold for 180+ days

### How to Archive

```bash
# Archive an agent
node scripts/taxonomy/archive-agent.js \
  --agent "backend-api-v1" \
  --reason "Superseded by v2" \
  --replacement ".claude/active/agents/specialized/backend-api-v2.md"

# Archive a skill
node scripts/taxonomy/archive-skill.js \
  --skill "old-tdd-workflow" \
  --reason "Methodology updated" \
  --replacement ".claude/active/skills/development/tdd-london-swarm/"

# Archive a project
node scripts/taxonomy/archive-project.js \
  --project "12fa-delivery" \
  --status "completed" \
  --outcome "successful"
```

---

## 🔍 Search Strategies

### By Name (Fastest)
```bash
# Find agent
cat .claude/active/agents/INDEX.md | grep "coder"

# Find skill
cat .claude/active/skills/SKILLS-CATALOG.md | grep "functionality-audit"
```

### By Tag (Most Flexible)
```bash
# Find all backend-related docs
cat docs/search/by-tag/backend.md

# Find all testing-related docs
cat docs/search/by-tag/testing.md
```

### By Category
```bash
# Active projects
ls docs/active/projects/

# Archived projects (2025 Q4)
ls docs/archive/projects/2025/Q4-projects/
```

### Full-Text Search
```bash
# Search all documentation
grep -r "pattern" docs/

# Search agent definitions
grep -r "backend-dev" .claude/active/agents/
```

---

## 🚀 Common Operations

### Create New Agent
```bash
# 1. Create definition file
touch .claude/active/agents/specialized/new-agent.md

# 2. Add frontmatter with metadata
# (see template in migration plan)

# 3. Update agent registry
node scripts/taxonomy/update-agent-registry.js

# 4. Generate indexes
node scripts/taxonomy/generate-indexes.js --target .claude/active/agents
```

### Create New Skill
```bash
# 1. Create skill directory
mkdir -p .claude/active/skills/development/new-skill

# 2. Add skill.md with frontmatter
touch .claude/active/skills/development/new-skill/skill.md

# 3. Update skills catalog
node scripts/taxonomy/update-skills-catalog.js

# 4. Generate indexes
node scripts/taxonomy/generate-indexes.js --target .claude/active/skills
```

### Start New Project
```bash
# 1. Create project directory
mkdir -p docs/active/projects/new-project

# 2. Use project template
cp -r docs/reference/templates/standard-project/* docs/active/projects/new-project/

# 3. Update project index
node scripts/taxonomy/update-project-index.js

# 4. Initialize project tracking
git add docs/active/projects/new-project/
git commit -m "docs: Initialize new-project"
```

### Complete and Archive Project
```bash
# 1. Archive project
node scripts/taxonomy/archive-project.js \
  --project "completed-project" \
  --status "completed" \
  --outcome "successful"

# 2. Update indexes
node scripts/taxonomy/generate-doc-indexes.js

# 3. Commit changes
git add docs/
git commit -m "docs: Archive completed-project (successful)"
```

---

## ⚠️ Common Pitfalls

### ❌ DON'T
- Store working files in root directories
- Archive reference material (update in place instead)
- Manually delete runtime data (use retention policies)
- Create custom directory structures (use taxonomy)
- Skip frontmatter metadata
- Archive without reason/replacement info
- Break existing file references

### ✅ DO
- Use appropriate subdirectories
- Update indexes after changes
- Add complete metadata (frontmatter)
- Document archival reasons
- Check for broken links
- Use migration scripts
- Backup before major changes

---

## 🛠️ Migration Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `create-structure.js` | Create directory structure | `node scripts/taxonomy/create-structure.js` |
| `inventory-files.js` | Generate file inventory | `node scripts/taxonomy/inventory-files.js` |
| `migrate-runtime-data.js` | Migrate runtime data | `node scripts/taxonomy/migrate-runtime-data.js --execute` |
| `generate-indexes.js` | Generate all indexes | `node scripts/taxonomy/generate-indexes.js` |
| `archive-agent.js` | Archive an agent | `node scripts/taxonomy/archive-agent.js --agent "name"` |
| `archive-skill.js` | Archive a skill | `node scripts/taxonomy/archive-skill.js --skill "name"` |
| `archive-project.js` | Archive a project | `node scripts/taxonomy/archive-project.js --project "name"` |
| `scheduled-archival.js` | Daily retention cleanup | `node scripts/taxonomy/scheduled-archival.js --now` |
| `weekly-maintenance.js` | Weekly maintenance | `node scripts/taxonomy/weekly-maintenance.js` |
| `validate-migration.js` | Validate migration | `node scripts/taxonomy/validate-migration.js` |

---

## 📊 Current State vs Target State

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| `.claude/projects` size | 560MB | <100MB | 82% reduction |
| Root-level docs files | 196 | <20 | 90% reduction |
| Navigation depth | Manual search | 5 clicks max | Instant navigation |
| Runtime cleanup | Manual | Automated | 100% automation |
| Archive retrieval | Difficult | <30 seconds | 95% faster |
| Document discovery | No structure | Tag-based | Systematic search |

---

## 🎯 Success Criteria

Migration is successful when:

- ✅ All files present (inventory match)
- ✅ No broken links (validation passed)
- ✅ Indexes complete and accurate
- ✅ Retention policies active
- ✅ Search functional (tag-based)
- ✅ Cross-references valid
- ✅ Automation scheduled
- ✅ Navigation <5 clicks to any doc

---

## 📞 Quick Help

**Find something?**
1. Start with master index: `docs/INDEX.md`
2. Use category indexes for specifics
3. Search by tag: `docs/search/by-tag/`
4. Full-text search as fallback

**Archive something?**
1. Check archival criteria
2. Use appropriate archive script
3. Update indexes
4. Commit with descriptive message

**Create something?**
1. Use appropriate active/ subdirectory
2. Follow naming conventions
3. Add complete metadata
4. Update indexes
5. Test cross-references

---

**For Complete Details**: See [TAXONOMY-REORGANIZATION-PLAN.md](TAXONOMY-REORGANIZATION-PLAN.md)
