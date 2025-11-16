# Taxonomy Visual Summary

**Date**: 2025-11-09
**Purpose**: Visual overview of new directory structure
**Related**: [TAXONOMY-REORGANIZATION-PLAN.md](TAXONOMY-REORGANIZATION-PLAN.md) | [TAXONOMY-QUICK-REFERENCE.md](TAXONOMY-QUICK-REFERENCE.md)

---

## 🎯 The 3-Tier Temporal Taxonomy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CLAUDE CODE ECOSYSTEM                            │
│                                                                     │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │
│  │    ACTIVE     │  │    ARCHIVE    │  │    RUNTIME    │          │
│  │               │  │               │  │               │          │
│  │  Current Work │  │  Historical   │  │  Operational  │          │
│  │  Manual Mgmt  │  │  Manual Arch  │  │  Auto-Cleanup │          │
│  │  Never Auto-  │  │  Searchable   │  │  Retention    │          │
│  │   Archived    │  │  Preserved    │  │  Policies     │          │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘          │
│          │                  │                  │                  │
│          └──────────────────┴──────────────────┘                  │
│                             │                                     │
│                    Coordinated Workflow                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Directory Structure Map

### `.claude/` - Configuration & Runtime

```
.claude/
│
├─ 🟢 active/                    Current Working State
│  ├─ agents/                   [131 agents organized by role]
│  │  ├─ core/                  8 foundational agents
│  │  ├─ specialized/           45 domain specialists
│  │  ├─ research/              4 Deep Research SOP
│  │  └─ meta/                  15 orchestrators
│  │
│  ├─ skills/                   [71 skills organized by function]
│  │  ├─ development/           18 dev skills
│  │  ├─ quality/               12 testing/audit
│  │  ├─ research/              9 research skills
│  │  ├─ infrastructure/        8 DevOps skills
│  │  └─ meta/                  7 meta-skills
│  │
│  ├─ commands/                 Slash command definitions
│  ├─ config/                   Active configuration
│  └─ artifacts/                Working research/analysis
│
├─ 🔵 archive/                   Historical Preservation
│  ├─ 2024/                     Year-based
│  │  ├─ agents/
│  │  ├─ skills/
│  │  ├─ commands/
│  │  └─ artifacts/
│  │
│  └─ 2025/                     Quarter-based
│     ├─ Q1/
│     ├─ Q2/
│     ├─ Q3/
│     └─ Q4/
│
└─ 🔴 runtime/                   Operational Data (Auto-Managed)
   ├─ logs/                     [Retention: 7-90 days]
   │  ├─ error/                 30-day retention
   │  ├─ debug/                 7-day retention
   │  └─ audit/                 90-day retention
   │
   ├─ history/                  Execution history
   ├─ cache/                    [TTL: 24 hours]
   │  ├─ shell-snapshots/       7-day retention
   │  └─ file-history/          30-day retention
   │
   └─ todos/                    Active TODO tracking
      ├─ current.json
      └─ completed/             30-day retention
```

### `docs/` - Documentation & Knowledge

```
docs/
│
├─ 🟢 active/                    Current Projects & Work
│  ├─ projects/                 [Active projects]
│  │  ├─ project-1/
│  │  │  ├─ README.md
│  │  │  ├─ architecture/
│  │  │  ├─ implementation/
│  │  │  ├─ testing/
│  │  │  └─ deployment/
│  │  └─ INDEX.md              Active projects list
│  │
│  ├─ workflows/                [Active workflows]
│  │  ├─ development/
│  │  ├─ testing/
│  │  └─ deployment/
│  │
│  └─ integration-plans/        [Active integrations]
│     ├─ mcp/
│     └─ external-tools/
│
├─ 🔵 archive/                   Completed Documentation
│  ├─ projects/                 [By year & quarter]
│  │  ├─ 2024/
│  │  │  ├─ Q1-projects/
│  │  │  ├─ Q2-projects/
│  │  │  ├─ Q3-projects/
│  │  │  └─ Q4-projects/
│  │  └─ 2025/
│  │     ├─ Q1-projects/       Example: 12fa-delivery
│  │     ├─ Q2-projects/
│  │     ├─ Q3-projects/
│  │     └─ Q4-projects/
│  │
│  ├─ research/                 [Completed research]
│  │  ├─ machine-learning/
│  │  ├─ system-architecture/
│  │  └─ security/
│  │
│  └─ experiments/              [Experimental results]
│     ├─ successful/
│     └─ failed/               Lessons learned
│
└─ 🟡 reference/                 Timeless Knowledge (Never Archived)
   ├─ agent-library/            [Agent reference docs]
   │  ├─ core-agents/
   │  ├─ specialized-agents/
   │  ├─ AGENT-REGISTRY.md     Complete registry
   │  └─ agent-creation/
   │
   ├─ skill-library/            [Skill reference docs]
   │  ├─ development/
   │  ├─ quality/
   │  ├─ research/
   │  ├─ SKILLS-CATALOG.md     Complete catalog
   │  └─ skill-creation/
   │
   ├─ architecture/             [Architecture reference]
   │  ├─ patterns/
   │  ├─ decisions/            ADRs
   │  └─ diagrams/
   │
   ├─ methodology/              [Dev methodologies]
   │  ├─ sparc/
   │  ├─ deep-research-sop/
   │  └─ tdd/
   │
   └─ tools/                    [Tool documentation]
      ├─ mcp-servers/
      ├─ claude-code/
      └─ external-tools/
```

---

## 🔄 Data Flow & Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONTENT LIFECYCLE                           │
└─────────────────────────────────────────────────────────────────┘

   Creation          Active Use         Completion        Archive
      │                  │                   │                │
      ▼                  ▼                   ▼                ▼
┌──────────┐      ┌──────────┐       ┌──────────┐     ┌──────────┐
│  Create  │─────▶│  Active  │──────▶│ Complete │────▶│ Archive  │
│   New    │      │   Work   │       │  Review  │     │  Store   │
└──────────┘      └──────────┘       └──────────┘     └──────────┘
                       │                                     │
                       │                                     │
                       ▼                                     ▼
                  ┌──────────┐                        ┌──────────┐
                  │ Runtime  │                        │Reference │
                  │  Data    │                        │ Material │
                  └──────────┘                        └──────────┘
                       │                                     │
                       ▼                                     │
                  ┌──────────┐                              │
                  │ Auto-    │                              │
                  │ Cleanup  │                              │
                  └──────────┘                              │
                       │                                     │
                       └─────────────────────────────────────┘
                              (Never Archived)

Legend:
─────▶  Manual workflow
┄┄┄┄▶  Automated workflow
```

### Specific Workflows

#### Agent Lifecycle
```
1. Create → .claude/active/agents/{category}/new-agent.md
2. Test & Refine → Update in place
3. Deprecate → Move to .claude/archive/{year}/agents/
4. Replace → Update references, add metadata
```

#### Project Lifecycle
```
1. Initialize → docs/active/projects/new-project/
2. Develop → Update documentation in place
3. Complete → Review & validate
4. Archive → Move to docs/archive/projects/{year}/{quarter}/
```

#### Runtime Data Lifecycle
```
1. Generate → .claude/runtime/{category}/
2. Age → Check against retention policy
3. Expire → Auto-delete or compress & archive
4. Clean → Remove from runtime storage
```

---

## 📊 Retention Policy Timeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   RETENTION POLICIES                            │
└─────────────────────────────────────────────────────────────────┘

Cache (24h)          ████ DELETE
Debug logs (7d)      ████████████████ DELETE
Shell snapshots (7d) ████████████████ DELETE
Error logs (30d)     ████████████████████████████████████████████████ COMPRESS & ARCHIVE
File history (30d)   ████████████████████████████████████████████████ COMPRESS & ARCHIVE
Completed TODOs (30d)████████████████████████████████████████████████ ARCHIVE TO JSON
Audit logs (90d)     ████████████████████████████████████████████████████████████████████████████████████████████ COMPRESS & ARCHIVE

│────│────│────│────│────│────│────│────│────│────│────│────│────│
0    7   14   21   28   35   42   49   56   63   70   77   84   90 days

Automated Daily (2 AM):
- Cache cleanup
- Debug log deletion
- Shell snapshot deletion

Automated Weekly:
- Error log archival
- File history archival
- Completed TODO archival

Automated Monthly:
- Audit log archival
- Archive compression
- Storage optimization
```

---

## 🔍 Navigation Pathways

### Finding Content - Decision Tree

```
                     ┌─────────────────┐
                     │  Need to find   │
                     │   something?    │
                     └────────┬────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
         ┌────▼────┐                    ┌────▼────┐
         │ Current │                    │Historical│
         │ Content │                    │ Content  │
         └────┬────┘                    └────┬────┘
              │                               │
    ┌─────────┴─────────┐          ┌─────────┴─────────┐
    │                   │          │                   │
┌───▼───┐          ┌───▼───┐  ┌───▼───┐          ┌───▼───┐
│Agent/ │          │Project│  │Archive│          │Search │
│ Skill │          │  Doc  │  │ Index │          │By Tag │
└───┬───┘          └───┬───┘  └───┬───┘          └───┬───┘
    │                  │          │                  │
    ▼                  ▼          ▼                  ▼
.claude/          docs/        docs/           docs/search/
 active/          active/      archive/        by-tag/
 {category}/      projects/    INDEX.md        {tag}.md
 INDEX.md         INDEX.md
```

### Index File Hierarchy

```
                    docs/INDEX.md
                    (Master Index)
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
  Active Index      Archive Index    Reference Index
        │                 │                 │
    ┌───┴───┐         ┌───┴───┐        ┌───┴───┐
    │       │         │       │        │       │
    ▼       ▼         ▼       ▼        ▼       ▼
Projects Workflows  2024   2025    Agents   Skills
 INDEX    INDEX     INDEX  INDEX    INDEX    CATALOG
```

---

## 📈 Migration Progress Tracker

```
┌─────────────────────────────────────────────────────────────────┐
│              14-DAY MIGRATION TIMELINE                          │
└─────────────────────────────────────────────────────────────────┘

Phase 1: Preparation
Day 1  ████████████████████ 100%  ✅ Directory structure, backup

Phase 2: Runtime Data
Day 2  ████████████████████ 100%  ⏳ Migrate runtime, apply retention

Phase 3: Active Content
Day 3  ██████████░░░░░░░░░░  50%  ⏳ Categorize agents
Day 4  ████████████████████ 100%  ⏳ Migrate skills/commands

Phase 4: Documentation
Day 5  ░░░░░░░░░░░░░░░░░░░░   0%  ⏸️  Organize docs
Day 6  ░░░░░░░░░░░░░░░░░░░░   0%  ⏸️  Manual review
Day 7  ░░░░░░░░░░░░░░░░░░░░   0%  ⏸️  Generate indexes

Phase 5: Archive
Day 8  ░░░░░░░░░░░░░░░░░░░░   0%  ⏸️  Archive old content
Day 9  ░░░░░░░░░░░░░░░░░░░░   0%  ⏸️  Add metadata

Phase 6: Reference
Day 10 ░░░░░░░░░░░░░░░░░░░░   0%  ⏸️  Consolidate reference

Phase 7: Validation
Day 11 ░░░░░░░░░░░░░░░░░░░░   0%  ⏸️  Validate migration
Day 12 ░░░░░░░░░░░░░░░░░░░░   0%  ⏸️  Cleanup

Phase 8: Automation
Day 13 ░░░░░░░░░░░░░░░░░░░░   0%  ⏸️  Install automation
Day 14 ░░░░░░░░░░░░░░░░░░░░   0%  ⏸️  Final testing

Legend:
████ Completed  ░░░░ Pending  ⏳ In Progress  ⏸️ Not Started
```

---

## 🎯 Impact Visualization

### Before Migration

```
docs/ (ROOT)
├─ file1.md
├─ file2.md
├─ file3.md
├─ ... (196 files!)       ⚠️ Disorganized
├─ file196.md
└─ subdirs/               ⚠️ Mixed structure
   └─ more-files/

.claude/
├─ projects/ (560MB!)     ⚠️ Old history
├─ debug/ (35MB)          ⚠️ No cleanup
└─ logs/                  ⚠️ Manual deletion

Navigation: ❌ Manual search required
Discovery: ❌ No systematic method
Cleanup: ❌ Manual effort
```

### After Migration

```
docs/
├─ INDEX.md                       ✅ Master navigation
├─ active/
│  ├─ projects/INDEX.md           ✅ Current work
│  └─ workflows/INDEX.md
├─ archive/
│  ├─ INDEX.md                    ✅ Historical search
│  └─ 2025/Q4-projects/
└─ reference/
   ├─ agent-library/INDEX.md      ✅ Timeless knowledge
   └─ skill-library/CATALOG.md

.claude/
├─ active/                        ✅ Clear structure
│  ├─ agents/INDEX.md
│  └─ skills/CATALOG.md
├─ archive/                       ✅ Historical preservation
│  └─ 2025/Q4/
└─ runtime/                       ✅ Auto-managed (<100MB)
   ├─ logs/ (7-90d retention)
   └─ cache/ (24h TTL)

Navigation: ✅ 5 clicks max to any doc
Discovery: ✅ Tag-based search + indexes
Cleanup: ✅ 100% automated
```

### Metrics Improvement

```
┌─────────────────────────────────────────────────────────────┐
│                    IMPROVEMENT METRICS                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Storage Reduction                                          │
│  Before: ████████████████████████████ 560MB                │
│  After:  ██████░░░░░░░░░░░░░░░░░░░░░░ 100MB                │
│  Saved:  ████████████████████ 82% reduction                │
│                                                             │
│  Root-Level Files                                           │
│  Before: ████████████████████████████ 196 files            │
│  After:  ██░░░░░░░░░░░░░░░░░░░░░░░░░░  20 files            │
│  Reduced: ██████████████████████ 90% cleaner               │
│                                                             │
│  Navigation Efficiency                                      │
│  Before: ████████████████████████████ Manual search        │
│  After:  ██░░░░░░░░░░░░░░░░░░░░░░░░░░  5 clicks max        │
│  Faster: ██████████████████████ 95% improvement            │
│                                                             │
│  Discovery Time                                             │
│  Before: ████████████████████████████ Minutes              │
│  After:  ██░░░░░░░░░░░░░░░░░░░░░░░░░░  Seconds             │
│  Faster: ██████████████████████ 80% improvement            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛡️ Safety & Validation

### Pre-Migration Checklist
```
☐ Full backup created (claude-backup-YYYYMMDD.tar.gz)
☐ Backup verified (file count matches)
☐ Backup stored externally (safe location)
☐ Directory structure created (.claude/active, docs/active, etc.)
☐ Migration scripts tested (dry-run mode)
☐ Team notified (if applicable)
☐ Rollback plan ready
```

### Post-Migration Validation
```
☐ File count matches inventory
☐ No files missing (diff check)
☐ Links validated (no broken references)
☐ Indexes complete (all categories covered)
☐ Retention policies active (scheduled tasks running)
☐ Search functional (tag-based working)
☐ Cross-references valid (agent↔skill↔command)
☐ User acceptance testing passed
```

### Rollback Triggers
```
⚠️ Stop migration if:
  - File loss detected (inventory mismatch)
  - Broken references >10% of links
  - Index generation fails
  - Automation setup fails
  - User acceptance test fails

→ Execute rollback:
  1. pkill -f "taxonomy"
  2. Restore from backup
  3. Verify restoration
  4. Investigate failure
  5. Fix and retry
```

---

## 📚 Documentation Reference

| Document | Purpose | Audience |
|----------|---------|----------|
| **TAXONOMY-REORGANIZATION-PLAN.md** | Complete implementation plan (70 pages) | Implementers |
| **TAXONOMY-QUICK-REFERENCE.md** | Fast lookup guide (10 pages) | Daily users |
| **TAXONOMY-VISUAL-SUMMARY.md** | Visual overview (this doc) | Stakeholders |
| **Migration Scripts** | Automated execution | Scripts |
| **Index Files** | Navigation | All users |

---

## 🚀 Next Actions

### Immediate (Today)
1. ✅ Review this plan
2. ✅ Approve migration approach
3. ⏳ Create directory structure
4. ⏳ Generate file inventory
5. ⏳ Create backup

### Short-Term (This Week)
1. ⏸️ Execute Phase 1-3 (runtime & active content)
2. ⏸️ Generate initial indexes
3. ⏸️ Validate structure
4. ⏸️ Test navigation

### Medium-Term (Next 2 Weeks)
1. ⏸️ Complete documentation migration
2. ⏸️ Archive historical content
3. ⏸️ Setup automation
4. ⏸️ Final validation

---

**Document Status**: Implementation-Ready Visual Overview
**Related Plans**:
- [TAXONOMY-REORGANIZATION-PLAN.md](TAXONOMY-REORGANIZATION-PLAN.md) - Detailed plan
- [TAXONOMY-QUICK-REFERENCE.md](TAXONOMY-QUICK-REFERENCE.md) - Quick lookup
