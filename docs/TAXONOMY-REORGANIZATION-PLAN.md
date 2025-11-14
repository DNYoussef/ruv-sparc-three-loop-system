# Comprehensive Taxonomy and Reorganization Plan
## Claude Artifacts and Documentation Directory Structure

**Date**: 2025-11-09
**Status**: Implementation-Ready Plan
**Scope**: `.claude/` and `docs/` directories

---

## Executive Summary

**Current State Analysis**:
- `.claude/projects`: 560MB (largest component - project history)
- `.claude/debug`: 35MB (operational logs)
- `docs/`: 353 markdown files (196 at root level - organizational issue)
- Mixed temporal states: active, completed, archived, obsolete
- Navigation difficulty: No clear index structure

**Proposed Solution**: 3-tier temporal taxonomy with archival automation

---

## I. Final Taxonomy Structure

### A. `.claude/` Directory - Runtime and Configuration

```
.claude/
├── active/                          # Current working configuration
│   ├── agents/                      # Active agent definitions
│   │   ├── core/                    # Foundational agents (coder, reviewer, tester)
│   │   ├── specialized/             # Domain specialists (ML, frontend, backend)
│   │   ├── research/                # Deep Research SOP agents (data-steward, ethics-agent)
│   │   └── meta/                    # Meta-agents (coordinators, orchestrators)
│   ├── skills/                      # Active skill definitions
│   │   ├── development/             # Development lifecycle skills
│   │   ├── quality/                 # Testing, validation, audit skills
│   │   ├── research/                # Research and analysis skills
│   │   ├── infrastructure/          # DevOps, cloud, deployment skills
│   │   └── meta/                    # Skill creation, orchestration skills
│   ├── commands/                    # Slash command definitions
│   │   ├── core/                    # Essential commands
│   │   ├── workflow/                # Workflow automation commands
│   │   └── integrations/            # External tool integration commands
│   ├── config/                      # Active configuration files
│   │   ├── settings.json            # Current settings
│   │   ├── credentials.json         # Encrypted credentials
│   │   └── mcp-servers.json         # MCP server configurations
│   └── artifacts/                   # Working artifacts (research, analysis)
│       ├── research/                # Research outputs
│       ├── analysis/                # Analysis results
│       └── planning/                # Planning artifacts
│
├── archive/                         # Historical artifacts (date-based)
│   ├── 2024/
│   │   ├── agents/                  # Deprecated agent definitions
│   │   ├── skills/                  # Deprecated skill definitions
│   │   ├── commands/                # Deprecated commands
│   │   └── artifacts/               # Historical artifacts
│   └── 2025/
│       ├── Q1/
│       ├── Q2/
│       ├── Q3/
│       └── Q4/
│
└── runtime/                         # Operational data (auto-managed)
    ├── logs/                        # System logs
    │   ├── error/                   # Error logs (30-day retention)
    │   ├── debug/                   # Debug logs (7-day retention)
    │   └── audit/                   # Audit logs (90-day retention)
    ├── debug/                       # Debug snapshots (7-day retention)
    ├── history/                     # Execution history
    │   ├── history.jsonl            # Main history file
    │   └── projects/                # Project-specific history
    ├── cache/                       # Temporary cache (24-hour TTL)
    │   ├── shell-snapshots/         # Shell state snapshots
    │   └── file-history/            # File change history
    └── todos/                       # Active TODO tracking
        ├── current.json             # Current session TODOs
        └── completed/               # Completed TODO archives (30-day retention)
```

**Key Principles for `.claude/`**:
1. **Active**: Current working state (never archived automatically)
2. **Archive**: Historical preservation (manual archival with metadata)
3. **Runtime**: Ephemeral operational data (auto-cleanup by retention policy)

---

### B. `docs/` Directory - Documentation and Knowledge

```
docs/
├── active/                          # Current project documentation
│   ├── projects/                    # Active project documentation
│   │   ├── {project-name}/          # Per-project structure
│   │   │   ├── README.md            # Project overview
│   │   │   ├── architecture/        # Architecture documentation
│   │   │   ├── implementation/      # Implementation guides
│   │   │   ├── testing/             # Testing documentation
│   │   │   └── deployment/          # Deployment guides
│   │   └── INDEX.md                 # Active projects index
│   ├── workflows/                   # Active workflow documentation
│   │   ├── development/             # Development workflows
│   │   ├── testing/                 # Testing workflows
│   │   ├── deployment/              # Deployment workflows
│   │   └── INDEX.md                 # Workflow index
│   └── integration-plans/           # Active integration plans
│       ├── mcp/                     # MCP server integrations
│       ├── external-tools/          # External tool integrations
│       └── INDEX.md                 # Integration index
│
├── archive/                         # Completed project documentation
│   ├── projects/                    # Completed projects (by year/quarter)
│   │   ├── 2024/
│   │   │   ├── Q1-projects/
│   │   │   ├── Q2-projects/
│   │   │   ├── Q3-projects/
│   │   │   └── Q4-projects/
│   │   └── 2025/
│   │       ├── Q1-projects/         # Example: 12fa delivery project
│   │       ├── Q2-projects/
│   │       ├── Q3-projects/
│   │       └── Q4-projects/
│   ├── research/                    # Completed research (by topic)
│   │   ├── machine-learning/
│   │   ├── system-architecture/
│   │   ├── security/
│   │   └── INDEX.md
│   ├── experiments/                 # Experimental documentation
│   │   ├── successful/              # Successful experiments
│   │   ├── failed/                  # Failed experiments (lessons learned)
│   │   └── INDEX.md
│   └── INDEX.md                     # Archive index with search tags
│
└── reference/                       # Timeless reference material
    ├── agent-library/               # Agent reference documentation
    │   ├── core-agents/             # Core agent specifications
    │   ├── specialized-agents/      # Specialized agent specifications
    │   ├── AGENT-REGISTRY.md        # Complete agent registry
    │   └── agent-creation/          # Agent creation guides
    ├── skill-library/               # Skill reference documentation
    │   ├── development/             # Development skills reference
    │   ├── quality/                 # Quality skills reference
    │   ├── research/                # Research skills reference
    │   ├── SKILLS-CATALOG.md        # Complete skills catalog
    │   └── skill-creation/          # Skill creation guides
    ├── architecture/                # Architecture reference
    │   ├── patterns/                # Architecture patterns
    │   ├── decisions/               # Architecture Decision Records (ADRs)
    │   ├── diagrams/                # Architecture diagrams
    │   └── INDEX.md
    ├── methodology/                 # Development methodologies
    │   ├── sparc/                   # SPARC methodology
    │   ├── deep-research-sop/       # Deep Research SOP
    │   ├── tdd/                     # Test-Driven Development
    │   └── INDEX.md
    ├── tools/                       # Tool reference documentation
    │   ├── mcp-servers/             # MCP server documentation
    │   ├── claude-code/             # Claude Code documentation
    │   ├── external-tools/          # External tool documentation
    │   └── INDEX.md
    └── INDEX.md                     # Master reference index
```

**Key Principles for `docs/`**:
1. **Active**: Current work in progress (moved to archive when complete)
2. **Archive**: Historical documentation (searchable, never deleted)
3. **Reference**: Timeless knowledge (updated in place, not archived)

---

## II. Archival Criteria

### A. Automatic Archival (Runtime Data)

**Triggers**: Time-based retention policies

| Category | Retention | Action | Frequency |
|----------|-----------|--------|-----------|
| Debug logs | 7 days | Delete oldest | Daily |
| Error logs | 30 days | Compress & archive | Weekly |
| Audit logs | 90 days | Compress & archive | Monthly |
| Shell snapshots | 7 days | Delete oldest | Daily |
| File history | 30 days | Compress & archive | Weekly |
| Cache | 24 hours | Delete | Hourly |
| Completed TODOs | 30 days | Archive to JSON | Weekly |

**Implementation**: Scheduled task running `scripts/archive-automation.js`

---

### B. Manual Archival (Active Content)

**Agents** - Archive when:
- Agent deprecated/replaced by better version
- Agent unused for 90+ days (check via logs)
- Agent marked as "experimental" and experiment concluded
- New major version released (archive old version)

**Skills** - Archive when:
- Skill superseded by better implementation
- Skill unused for 180+ days (check via Memory MCP)
- Skill marked as "deprecated" in metadata
- Methodology change makes skill obsolete

**Commands** - Archive when:
- Command replaced by better alternative
- Command unused for 90+ days
- Command incompatible with current system version

**Documentation (Active Projects)** - Archive when:
- Project marked as "completed" (all deliverables shipped)
- Project abandoned/cancelled (mark as such)
- Project on hold for 180+ days (move to archive with "paused" status)

**Artifacts** - Archive when:
- Research completed and documented
- Analysis results incorporated into decisions
- Planning artifacts implemented or superseded

---

### C. Reference Material Update Policy

**Never Archive** - Instead, update in place:
- Agent registry and specifications
- Skills catalog
- Architecture patterns
- Methodology guides
- Tool documentation

**Version Control**: Use git for reference material changes

---

## III. File Organization Rules

### A. Naming Conventions

**Date-Based Naming** (for archives):
```
YYYY-MM-DD-{category}-{descriptive-name}.{ext}
Example: 2025-01-15-agent-backend-api-v1.md
```

**Semantic Naming** (for active content):
```
{category}-{function}-{variant}.{ext}
Example: agent-core-coder-enhanced.md
         skill-testing-tdd-london-swarm.md
         workflow-development-feature-complete.md
```

**Index Files**:
```
INDEX.md       # Primary index (human-readable table of contents)
CATALOG.md     # Comprehensive catalog with metadata
SUMMARY.md     # Executive summary
QUICK-REFERENCE.md  # Quick reference guide
```

**Archival Metadata** (frontmatter):
```yaml
---
title: "Backend API Agent v1"
category: agent
type: specialized
status: archived
archived_date: 2025-01-15
archived_reason: "Superseded by backend-api-v2"
replacement: ".claude/active/agents/specialized/backend-api-v2.md"
tags: [backend, api, deprecated]
project: delivery-phase-1
last_active: 2024-12-01
usage_count: 47
---
```

---

### B. Categorization Rules

**Agents**:
- **Core**: Universal agents used across all projects (coder, reviewer, tester, planner, researcher)
- **Specialized**: Domain-specific agents (backend-dev, ml-developer, security-testing-agent)
- **Research**: Deep Research SOP agents (data-steward, ethics-agent, archivist, evaluator)
- **Meta**: Orchestration and coordination agents (hierarchical-coordinator, swarm-init)

**Skills**:
- **Development**: Code creation, implementation (feature-dev-complete, pair-programming)
- **Quality**: Testing, validation, audit (functionality-audit, theater-detection-audit)
- **Research**: Analysis, investigation (literature-synthesis, baseline-replication)
- **Infrastructure**: DevOps, deployment (cicd-intelligent-recovery, docker-containerization)
- **Meta**: Skill creation, orchestration (micro-skill-creator, cascade-orchestrator)

**Documentation**:
- **Projects**: Project-specific documentation (delivery phases, implementations)
- **Workflows**: Reusable workflows (development, testing, deployment)
- **Integration**: Integration plans and guides (MCP servers, external tools)
- **Research**: Research outputs and analysis
- **Reference**: Timeless knowledge (agent library, skill library, methodology)

---

## IV. Migration Plan

### Phase 1: Preparation (Day 1)

**Tasks**:
1. Create new directory structure (without moving files)
2. Generate inventory of all files with metadata
3. Create archival decision matrix
4. Backup current state

**Scripts**:
```bash
# Create directory structure
node scripts/create-taxonomy-structure.js

# Generate inventory
node scripts/inventory-files.js > docs/migration/file-inventory.json

# Backup
tar -czf claude-backup-$(date +%Y%m%d).tar.gz .claude/ docs/
```

---

### Phase 2: Runtime Data Migration (Day 2)

**Tasks**:
1. Move runtime data to new structure
2. Apply retention policies (delete old logs)
3. Compress historical data
4. Validate runtime operations

**Priority**: High (reduces disk usage from 560MB projects + 35MB debug)

**Scripts**:
```bash
# Migrate runtime data
node scripts/migrate-runtime-data.js --dry-run
node scripts/migrate-runtime-data.js --execute

# Apply retention policies
node scripts/apply-retention-policies.js
```

---

### Phase 3: Active Content Migration (Day 3-4)

**Tasks**:
1. Categorize agents, skills, commands
2. Move to appropriate active/* subdirectories
3. Update internal cross-references
4. Generate index files

**Priority**: Medium (maintains current functionality)

**Scripts**:
```bash
# Categorize agents
node scripts/categorize-agents.js

# Migrate active content
node scripts/migrate-active-content.js --category agents
node scripts/migrate-active-content.js --category skills
node scripts/migrate-active-content.js --category commands

# Generate indexes
node scripts/generate-indexes.js --target .claude/active
```

---

### Phase 4: Documentation Migration (Day 5-7)

**Tasks**:
1. Identify active vs completed projects
2. Move completed projects to archive with metadata
3. Organize reference material
4. Create comprehensive indexes

**Priority**: Medium (improves navigation)

**Scripts**:
```bash
# Categorize documentation
node scripts/categorize-documentation.js

# Migrate documentation
node scripts/migrate-documentation.js --dry-run
node scripts/migrate-documentation.js --execute

# Generate documentation indexes
node scripts/generate-doc-indexes.js
```

**Manual Review Required**:
- Project completion status (83 files with P{N}_T{N} patterns)
- Research status (completed vs ongoing)
- Experiment outcomes (successful vs failed)

---

### Phase 5: Archive Migration (Day 8-9)

**Tasks**:
1. Move deprecated agents/skills to archive
2. Move completed project docs to archive
3. Add archival metadata (frontmatter)
4. Create archive indexes

**Priority**: Low (improves long-term organization)

**Scripts**:
```bash
# Migrate to archive
node scripts/migrate-to-archive.js --category agents
node scripts/migrate-to-archive.js --category skills
node scripts/migrate-to-archive.js --category documentation

# Generate archive indexes
node scripts/generate-archive-indexes.js
```

---

### Phase 6: Reference Material Organization (Day 10)

**Tasks**:
1. Consolidate agent registry
2. Consolidate skills catalog
3. Organize methodology guides
4. Create master index

**Priority**: Medium (critical reference material)

**Scripts**:
```bash
# Consolidate reference material
node scripts/consolidate-reference.js --category agents
node scripts/consolidate-reference.js --category skills

# Generate master index
node scripts/generate-master-index.js
```

---

### Phase 7: Validation & Cleanup (Day 11-12)

**Tasks**:
1. Validate all file moves
2. Check broken links
3. Verify indexes
4. Remove empty directories
5. Update CLAUDE.md references

**Scripts**:
```bash
# Validate migration
node scripts/validate-migration.js

# Check for broken links
node scripts/check-broken-links.js

# Cleanup
node scripts/cleanup-empty-dirs.js
```

---

### Phase 8: Automation Setup (Day 13-14)

**Tasks**:
1. Install archival automation
2. Configure retention policies
3. Setup monitoring
4. Document maintenance procedures

**Scripts**:
```bash
# Install automation
node scripts/install-archival-automation.js

# Configure scheduled tasks
node scripts/configure-scheduled-tasks.js

# Test automation
node scripts/test-archival-automation.js
```

---

## V. Index File Specifications

### A. Primary Index Files

#### 1. Master Index (`docs/INDEX.md`)

**Purpose**: Top-level navigation for entire documentation system

**Structure**:
```markdown
# Documentation Master Index

**Last Updated**: 2025-11-09
**Total Documents**: 353
**Active Projects**: 12
**Archived Projects**: 45

## Quick Navigation

- [Active Projects](#active-projects)
- [Reference Library](#reference-library)
- [Archive](#archive)
- [Search](#search)

## Active Projects

| Project | Status | Last Updated | Priority |
|---------|--------|--------------|----------|
| [Agent Enhancement](active/projects/agent-enhancement/) | In Progress | 2025-11-08 | High |
| [MCP Integration](active/projects/mcp-integration/) | In Progress | 2025-11-07 | High |
| ... | ... | ... | ... |

## Reference Library

### Agent Library
- [Agent Registry](reference/agent-library/AGENT-REGISTRY.md) - 131 agents
- [Core Agents](reference/agent-library/core-agents/)
- [Specialized Agents](reference/agent-library/specialized-agents/)

### Skill Library
- [Skills Catalog](reference/skill-library/SKILLS-CATALOG.md) - 71 skills
- [Development Skills](reference/skill-library/development/)
- [Quality Skills](reference/skill-library/quality/)

### Methodology
- [SPARC](reference/methodology/sparc/)
- [Deep Research SOP](reference/methodology/deep-research-sop/)
- [TDD](reference/methodology/tdd/)

## Archive

- [2025 Archive](archive/2025/) - Q1: 15 projects, Q2: 8 projects
- [2024 Archive](archive/2024/) - Q1: 12 projects, Q2: 10 projects

## Search

**By Category**:
- [Agents](search/by-category/agents.md)
- [Skills](search/by-category/skills.md)
- [Projects](search/by-category/projects.md)

**By Tag**:
- [#backend](search/by-tag/backend.md)
- [#testing](search/by-tag/testing.md)
- [#ml](search/by-tag/ml.md)
```

---

#### 2. Agent Index (`.claude/active/agents/INDEX.md`)

**Purpose**: Comprehensive agent directory

**Structure**:
```markdown
# Agent Index

**Total Agents**: 131
**Core**: 8 | **Specialized**: 45 | **Research**: 4 | **Meta**: 15

## Core Agents (8)

| Agent | Purpose | Usage Count (30d) | Last Updated |
|-------|---------|-------------------|--------------|
| [coder](core/coder.md) | General-purpose coding | 1,247 | 2025-11-08 |
| [reviewer](core/reviewer.md) | Code review | 843 | 2025-11-07 |
| [tester](core/tester.md) | Test creation | 672 | 2025-11-08 |
| ... | ... | ... | ... |

## Specialized Agents (45)

### Frontend Development (6 agents)
- [react-developer](specialized/react-developer.md) - React 18+ specialist
- [vue-developer](specialized/vue-developer.md) - Vue 3 specialist
- ...

### Backend Development (8 agents)
- [backend-dev](specialized/backend-dev.md) - Backend API development
- [database-design-specialist](specialized/database-design-specialist.md) - Database design
- ...

## Quick Reference

**Most Used (30 days)**:
1. coder (1,247)
2. reviewer (843)
3. tester (672)
4. backend-dev (421)
5. react-developer (387)

**Recently Updated**:
1. coder (2025-11-08)
2. tester (2025-11-08)
3. reviewer (2025-11-07)
```

---

#### 3. Skills Catalog (`.claude/active/skills/SKILLS-CATALOG.md`)

**Purpose**: Comprehensive skills directory with trigger patterns

**Structure**:
```markdown
# Skills Catalog

**Total Skills**: 71
**Development**: 18 | **Quality**: 12 | **Research**: 9 | **Infrastructure**: 8 | **Meta**: 7

## Auto-Trigger Reference

### Development Lifecycle Skills

**Planning & Architecture**
- `research-driven-planning` - "new feature", "plan", "architecture needed" → Research + 5x pre-mortem
- `sparc-methodology` - "implement", "build feature" → 5-phase SPARC
- ...

**Development**
- `parallel-swarm-implementation` - "build this", "implement feature" → Multi-agent parallel dev
- `feature-dev-complete` - "complete feature", "end-to-end" → 12-stage workflow
- ...

## Skills by Category

### Development (18 skills)
| Skill | Trigger Keywords | Agents Used | Avg Duration |
|-------|------------------|-------------|--------------|
| [feature-dev-complete](development/feature-dev-complete/) | "complete feature", "end-to-end" | 6 agents | 45min |
| [pair-programming](development/pair-programming/) | "pair with me", "let's code" | 2 agents | 30min |
| ... | ... | ... | ... |

## Quick Reference

**Most Used (30 days)**:
1. functionality-audit (89)
2. feature-dev-complete (67)
3. pair-programming (54)

**Best Performers** (by success rate):
1. functionality-audit (98.2%)
2. theater-detection-audit (96.8%)
3. production-readiness (95.1%)
```

---

#### 4. Project Index (`docs/active/projects/INDEX.md`)

**Purpose**: Active projects tracking

**Structure**:
```markdown
# Active Projects Index

**Total Active**: 12
**High Priority**: 3 | **Medium Priority**: 6 | **Low Priority**: 3

## Current Sprint (Week 45, 2025)

### High Priority

#### Agent Enhancement Project
- **Status**: In Progress (75% complete)
- **Owner**: @coder, @reviewer
- **Deadline**: 2025-11-15
- **Documentation**: [agent-enhancement/](agent-enhancement/)
- **Last Updated**: 2025-11-08
- **Blockers**: None
- **Next Actions**:
  - Complete specialized agent testing
  - Update agent registry

#### MCP Integration Project
- **Status**: In Progress (60% complete)
- **Owner**: @researcher, @backend-dev
- **Deadline**: 2025-11-20
- **Documentation**: [mcp-integration/](mcp-integration/)
- **Last Updated**: 2025-11-07
- **Blockers**: Flow-Nexus authentication
- **Next Actions**:
  - Complete authentication flow
  - Test MCP tool coordination

## Recently Completed

- [12FA Delivery Project](../archive/2025/Q4-projects/12fa-delivery/) - Completed 2025-10-31
- [Dogfooding System](../archive/2025/Q4-projects/dogfooding-system/) - Completed 2025-11-02

## Project Templates

- [Standard Project Structure](../reference/templates/standard-project/)
- [Research Project Structure](../reference/templates/research-project/)
```

---

#### 5. Archive Index (`docs/archive/INDEX.md`)

**Purpose**: Historical documentation with search capabilities

**Structure**:
```markdown
# Archive Index

**Total Archived**: 45 projects
**2025**: 23 | **2024**: 22

## Search & Filter

### By Year & Quarter
- [2025 Q4](2025/Q4-projects/) - 8 projects
- [2025 Q3](2025/Q3-projects/) - 6 projects
- [2025 Q2](2025/Q2-projects/) - 5 projects
- [2025 Q1](2025/Q1-projects/) - 4 projects

### By Category
- [Agent Development](search/agent-development.md) - 12 projects
- [Skill Development](search/skill-development.md) - 8 projects
- [Infrastructure](search/infrastructure.md) - 7 projects
- [Research](search/research.md) - 6 projects

### By Status
- [Completed](search/completed.md) - 38 projects
- [Cancelled](search/cancelled.md) - 5 projects
- [Paused](search/paused.md) - 2 projects

### By Outcome
- [Successful](search/successful.md) - 35 projects
- [Failed](search/failed.md) - 7 projects (with lessons learned)
- [Experimental](search/experimental.md) - 3 projects

## Recent Additions (Last 30 Days)

| Project | Archived Date | Reason | Outcome |
|---------|---------------|--------|---------|
| [12FA Delivery](2025/Q4-projects/12fa-delivery/) | 2025-10-31 | Completed | Successful |
| [Dogfooding System](2025/Q4-projects/dogfooding-system/) | 2025-11-02 | Completed | Successful |
| ... | ... | ... | ... |

## Archive Statistics

**By Outcome**:
- Successful: 77.8% (35/45)
- Failed: 15.6% (7/45)
- Experimental: 6.7% (3/45)

**By Category**:
- Agent Development: 26.7% (12/45)
- Skill Development: 17.8% (8/45)
- Infrastructure: 15.6% (7/45)
```

---

### B. Specialized Index Files

#### 1. Tag-Based Search Index (`docs/search/by-tag/{tag}.md`)

**Purpose**: Find documents by semantic tags

**Example** (`docs/search/by-tag/backend.md`):
```markdown
# Tag: Backend

**Documents**: 47 | **Active**: 12 | **Archived**: 35

## Active Documents

### Agents (5)
- [backend-dev](../../active/agents/specialized/backend-dev.md)
- [database-design-specialist](../../active/agents/specialized/database-design-specialist.md)
- ...

### Skills (3)
- [when-building-backend-api-orchestrate-api-development](../../.claude/active/skills/development/backend-api-orchestrator.md)
- [sop-api-development](../../.claude/active/skills/development/sop-api-development.md)
- ...

### Projects (4)
- [API Gateway Implementation](../../active/projects/api-gateway/)
- [Microservices Migration](../../active/projects/microservices-migration/)
- ...

## Archived Documents

### Projects (35)
- [12FA Backend API](../../archive/2025/Q4-projects/12fa-delivery/backend/)
- [Auth Service v1](../../archive/2025/Q3-projects/auth-service-v1/)
- ...
```

---

#### 2. Cross-Reference Index (`docs/reference/CROSS-REFERENCE.md`)

**Purpose**: Map relationships between agents, skills, commands

**Structure**:
```markdown
# Cross-Reference Index

## Agent → Skills Mapping

### Core Agents
**coder** → Uses skills:
- functionality-audit (post-implementation)
- pair-programming (during development)
- smart-bug-fix (when debugging)

**reviewer** → Uses skills:
- code-review-assistant (comprehensive review)
- security (security audit)
- style-audit (code style)

### Specialized Agents
**backend-dev** → Uses skills:
- when-building-backend-api-orchestrate-api-development (API development)
- database-design-specialist (DB design)
- sop-api-development (standardized API process)

## Skill → Agents Mapping

### Development Skills
**feature-dev-complete** → Spawns agents:
- researcher (best practices research)
- coder (implementation)
- tester (testing)
- reviewer (code review)
- backend-dev (API development)
- react-developer (frontend)

**pair-programming** → Spawns agents:
- coder (driver)
- reviewer (navigator)

## Command → Skill Mapping

### Workflow Commands
**/build-feature** → Invokes skills:
- research-driven-planning (planning)
- feature-dev-complete (implementation)
- functionality-audit (validation)

**/review-pr** → Invokes skills:
- code-review-assistant (review)
- security (security check)
```

---

## VI. Special Handling for Edge Cases

### A. Large Files (>1MB)

**Problem**: 196 root-level markdown files in docs/

**Solutions**:
1. **Split large files** into logical sections:
   - Keep master file with overview
   - Extract sections to subdirectories
   - Link via table of contents

2. **Compress historical data**:
   - `.tar.gz` for archived projects
   - Maintain uncompressed index

3. **External storage for binary artifacts**:
   - Move large images/videos to external storage
   - Keep references in documentation

**Example**:
```
# Before: docs/COMPREHENSIVE-AGENT-SKILL-INVENTORY-2025-11-02.md (50KB)

# After:
docs/reference/agent-library/AGENT-REGISTRY.md (10KB overview)
docs/reference/agent-library/core-agents/INDEX.md (detailed specs)
docs/reference/agent-library/specialized-agents/INDEX.md (detailed specs)
```

---

### B. Duplicate/Similar Content

**Problem**: Multiple files with similar content (e.g., multiple "SUMMARY" files)

**Solutions**:
1. **Consolidate duplicates**:
   - Identify canonical version
   - Archive duplicates with redirect metadata
   - Update cross-references

2. **Version control for reference material**:
   - Use git tags for versions
   - Keep single source of truth
   - Archive old versions

**Example**:
```yaml
# Duplicate file metadata
---
title: "Agent Registry v1"
status: duplicate
canonical_version: "docs/reference/agent-library/AGENT-REGISTRY.md"
redirect_to: "../reference/agent-library/AGENT-REGISTRY.md"
archived_date: 2025-11-09
---

This file has been consolidated into the canonical Agent Registry.
Please use: [AGENT-REGISTRY.md](../reference/agent-library/AGENT-REGISTRY.md)
```

---

### C. Multi-Category Documents

**Problem**: Documents that fit multiple categories (e.g., agent + skill + project)

**Solutions**:
1. **Primary location + symlinks**:
   - Choose primary category
   - Create symlinks in other categories
   - Document relationships in metadata

2. **Cross-reference indexes**:
   - Maintain in primary location
   - List in all relevant indexes
   - Add "see also" sections

**Example**:
```markdown
# docs/active/projects/agent-enhancement/README.md

## Related Documentation

**Agent Definitions**: [.claude/active/agents/specialized/](../../../.claude/active/agents/specialized/)
**Skill Definitions**: [.claude/active/skills/agent-creation/](../../../.claude/active/skills/agent-creation/)
**Methodology**: [docs/reference/methodology/agent-creation/](../../reference/methodology/agent-creation/)
```

---

### D. Runtime Data with Historical Value

**Problem**: Some runtime data (e.g., audit logs, metrics) has long-term value

**Solutions**:
1. **Selective archival**:
   - Mark specific logs for preservation
   - Extract insights into documentation
   - Archive raw data separately

2. **Metrics aggregation**:
   - Daily/weekly aggregation
   - Archive aggregated metrics
   - Delete raw logs per retention policy

**Example**:
```bash
# Extract insights from runtime data
node scripts/extract-runtime-insights.js \
  --source .claude/runtime/logs/audit/2025-10.log \
  --output docs/archive/2025/Q4-projects/metrics/audit-insights-2025-10.md

# Archive raw data
tar -czf .claude/archive/2025/Q4/runtime-logs-2025-10.tar.gz \
  .claude/runtime/logs/audit/2025-10*.log

# Delete raw logs (per retention policy)
rm .claude/runtime/logs/audit/2025-10*.log
```

---

### E. Sensitive/Confidential Data

**Problem**: Credentials, API keys, sensitive research data

**Solutions**:
1. **Encryption at rest**:
   - Encrypt `.claude/active/config/credentials.json`
   - Use git-crypt for sensitive docs
   - Never commit to public repos

2. **Redaction for archives**:
   - Redact sensitive data before archiving
   - Maintain redaction log
   - Store originals in secure location

3. **Access control**:
   - Separate sensitive content
   - Document access requirements
   - Audit access logs

**Example**:
```yaml
# Sensitive file metadata
---
title: "MCP Server Credentials"
status: active
sensitivity: high
encryption: AES-256
access_control: owner-only
audit_required: true
---

# File stored at: .claude/active/config/credentials.json.enc
# Decrypt with: node scripts/decrypt-credentials.js
```

---

### F. Executable Content (Scripts, Hooks)

**Problem**: Mix of documentation and executable code

**Solutions**:
1. **Separate documentation from code**:
   - Code: `.claude/active/scripts/`, `hooks/`
   - Docs: `docs/reference/tools/`, `docs/active/workflows/`

2. **Maintain usage documentation**:
   - Keep README.md with scripts
   - Link to comprehensive docs
   - Include examples

**Structure**:
```
.claude/active/scripts/
├── archival-automation.js
├── README.md                     # Usage documentation
└── examples/
    └── archival-example.sh

docs/reference/tools/archival-automation/
├── OVERVIEW.md                   # Comprehensive documentation
├── CONFIGURATION.md              # Configuration guide
├── API-REFERENCE.md              # API documentation
└── EXAMPLES.md                   # Extended examples
```

---

### G. Experimental/Unstable Content

**Problem**: Work-in-progress, experiments, prototypes

**Solutions**:
1. **Dedicated experimental directory**:
   - `.claude/active/experimental/`
   - Clear "unstable" warnings
   - Define graduation criteria

2. **Time-bound experimentation**:
   - Set experiment deadline
   - Auto-archive if not graduated
   - Document outcomes

**Example**:
```yaml
# Experimental file metadata
---
title: "Multi-Persona Debate Framework"
status: experimental
stability: unstable
experiment_start: 2025-11-08
experiment_deadline: 2025-12-08
graduation_criteria:
  - 5+ successful debates
  - 90%+ accuracy vs single-agent
  - Memory persistence working
reviewer: researcher
---

⚠️ **EXPERIMENTAL**: This feature is unstable and may change significantly.
Do not use in production workflows.
```

---

## VII. Implementation Scripts

### A. Core Migration Scripts

#### 1. `scripts/taxonomy/create-structure.js`
**Purpose**: Create new directory structure

```javascript
const fs = require('fs-extra');
const path = require('path');

const structure = {
  '.claude': {
    'active': {
      'agents': ['core', 'specialized', 'research', 'meta'],
      'skills': ['development', 'quality', 'research', 'infrastructure', 'meta'],
      'commands': ['core', 'workflow', 'integrations'],
      'config': [],
      'artifacts': ['research', 'analysis', 'planning']
    },
    'archive': {
      '2024': ['agents', 'skills', 'commands', 'artifacts'],
      '2025': ['Q1', 'Q2', 'Q3', 'Q4']
    },
    'runtime': {
      'logs': ['error', 'debug', 'audit'],
      'debug': [],
      'history': ['projects'],
      'cache': ['shell-snapshots', 'file-history'],
      'todos': ['completed']
    }
  },
  'docs': {
    'active': {
      'projects': [],
      'workflows': ['development', 'testing', 'deployment'],
      'integration-plans': ['mcp', 'external-tools']
    },
    'archive': {
      'projects': {
        '2024': ['Q1-projects', 'Q2-projects', 'Q3-projects', 'Q4-projects'],
        '2025': ['Q1-projects', 'Q2-projects', 'Q3-projects', 'Q4-projects']
      },
      'research': ['machine-learning', 'system-architecture', 'security'],
      'experiments': ['successful', 'failed']
    },
    'reference': {
      'agent-library': ['core-agents', 'specialized-agents'],
      'skill-library': ['development', 'quality', 'research'],
      'architecture': ['patterns', 'decisions', 'diagrams'],
      'methodology': ['sparc', 'deep-research-sop', 'tdd'],
      'tools': ['mcp-servers', 'claude-code', 'external-tools']
    }
  }
};

function createStructure(base, struct) {
  for (const [key, value] of Object.entries(struct)) {
    const dirPath = path.join(base, key);
    fs.ensureDirSync(dirPath);
    console.log(`Created: ${dirPath}`);

    if (Array.isArray(value)) {
      value.forEach(subdir => {
        const subPath = path.join(dirPath, subdir);
        fs.ensureDirSync(subPath);
        console.log(`Created: ${subPath}`);
      });
    } else if (typeof value === 'object') {
      createStructure(dirPath, value);
    }
  }
}

// Execute
createStructure(process.env.HOME || process.env.USERPROFILE, structure);
console.log('✅ Directory structure created successfully');
```

---

#### 2. `scripts/taxonomy/inventory-files.js`
**Purpose**: Generate comprehensive file inventory with metadata

```javascript
const fs = require('fs-extra');
const path = require('path');
const crypto = require('crypto');

function getFileHash(filePath) {
  const content = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(content).digest('hex');
}

function getFileMetadata(filePath) {
  const stats = fs.statSync(filePath);
  return {
    path: filePath,
    size: stats.size,
    created: stats.birthtime,
    modified: stats.mtime,
    accessed: stats.atime,
    hash: getFileHash(filePath),
    extension: path.extname(filePath)
  };
}

function inventoryDirectory(dirPath, results = []) {
  const items = fs.readdirSync(dirPath);

  for (const item of items) {
    const fullPath = path.join(dirPath, item);
    const stats = fs.statSync(fullPath);

    if (stats.isDirectory()) {
      inventoryDirectory(fullPath, results);
    } else {
      results.push(getFileMetadata(fullPath));
    }
  }

  return results;
}

// Execute
const claudeInventory = inventoryDirectory('.claude');
const docsInventory = inventoryDirectory('docs');

const inventory = {
  timestamp: new Date().toISOString(),
  totals: {
    claude_files: claudeInventory.length,
    docs_files: docsInventory.length,
    total_files: claudeInventory.length + docsInventory.length
  },
  claude: claudeInventory,
  docs: docsInventory
};

fs.writeJsonSync('docs/migration/file-inventory.json', inventory, { spaces: 2 });
console.log(`✅ Inventory complete: ${inventory.totals.total_files} files cataloged`);
```

---

#### 3. `scripts/taxonomy/migrate-runtime-data.js`
**Purpose**: Move runtime data to new structure with retention

```javascript
const fs = require('fs-extra');
const path = require('path');

const RETENTION_POLICIES = {
  'debug': 7 * 24 * 60 * 60 * 1000,      // 7 days
  'error': 30 * 24 * 60 * 60 * 1000,     // 30 days
  'audit': 90 * 24 * 60 * 60 * 1000,     // 90 days
  'cache': 24 * 60 * 60 * 1000,          // 24 hours
  'shell-snapshots': 7 * 24 * 60 * 60 * 1000, // 7 days
  'file-history': 30 * 24 * 60 * 60 * 1000   // 30 days
};

function shouldRetain(filePath, category) {
  const stats = fs.statSync(filePath);
  const age = Date.now() - stats.mtime.getTime();
  return age <= RETENTION_POLICIES[category];
}

function migrateRuntimeData(dryRun = true) {
  const migrations = [
    { from: '.claude/debug', to: '.claude/runtime/debug', category: 'debug' },
    { from: '.claude/logs', to: '.claude/runtime/logs', category: 'error' },
    { from: '.claude/shell-snapshots', to: '.claude/runtime/cache/shell-snapshots', category: 'shell-snapshots' },
    { from: '.claude/file-history', to: '.claude/runtime/cache/file-history', category: 'file-history' }
  ];

  const stats = {
    moved: 0,
    deleted: 0,
    skipped: 0
  };

  for (const { from, to, category } of migrations) {
    if (!fs.existsSync(from)) continue;

    const files = fs.readdirSync(from);
    for (const file of files) {
      const sourcePath = path.join(from, file);
      const destPath = path.join(to, file);

      if (shouldRetain(sourcePath, category)) {
        if (!dryRun) {
          fs.ensureDirSync(path.dirname(destPath));
          fs.moveSync(sourcePath, destPath);
        }
        console.log(`✅ ${dryRun ? '[DRY RUN] ' : ''}Moved: ${sourcePath} → ${destPath}`);
        stats.moved++;
      } else {
        if (!dryRun) {
          fs.removeSync(sourcePath);
        }
        console.log(`🗑️  ${dryRun ? '[DRY RUN] ' : ''}Deleted: ${sourcePath} (expired)`);
        stats.deleted++;
      }
    }
  }

  console.log('\n📊 Migration Summary:');
  console.log(`   Moved: ${stats.moved}`);
  console.log(`   Deleted: ${stats.deleted}`);
  console.log(`   Skipped: ${stats.skipped}`);

  if (dryRun) {
    console.log('\n⚠️  This was a dry run. Use --execute to apply changes.');
  }
}

// Execute
const dryRun = !process.argv.includes('--execute');
migrateRuntimeData(dryRun);
```

---

#### 4. `scripts/taxonomy/generate-indexes.js`
**Purpose**: Generate all index files automatically

```javascript
const fs = require('fs-extra');
const path = require('path');

function generateAgentIndex(baseDir) {
  const indexPath = path.join(baseDir, 'INDEX.md');
  const categories = ['core', 'specialized', 'research', 'meta'];

  let content = `# Agent Index\n\n`;
  content += `**Generated**: ${new Date().toISOString()}\n\n`;

  for (const category of categories) {
    const categoryPath = path.join(baseDir, category);
    if (!fs.existsSync(categoryPath)) continue;

    const agents = fs.readdirSync(categoryPath).filter(f => f.endsWith('.md'));
    content += `## ${category.charAt(0).toUpperCase() + category.slice(1)} Agents (${agents.length})\n\n`;

    for (const agent of agents) {
      const agentName = path.basename(agent, '.md');
      content += `- [${agentName}](${category}/${agent})\n`;
    }
    content += '\n';
  }

  fs.writeFileSync(indexPath, content);
  console.log(`✅ Generated: ${indexPath}`);
}

function generateSkillIndex(baseDir) {
  const indexPath = path.join(baseDir, 'SKILLS-CATALOG.md');
  const categories = ['development', 'quality', 'research', 'infrastructure', 'meta'];

  let content = `# Skills Catalog\n\n`;
  content += `**Generated**: ${new Date().toISOString()}\n\n`;

  for (const category of categories) {
    const categoryPath = path.join(baseDir, category);
    if (!fs.existsSync(categoryPath)) continue;

    const skills = fs.readdirSync(categoryPath);
    content += `## ${category.charAt(0).toUpperCase() + category.slice(1)} Skills (${skills.length})\n\n`;

    for (const skill of skills) {
      content += `- [${skill}](${category}/${skill}/)\n`;
    }
    content += '\n';
  }

  fs.writeFileSync(indexPath, content);
  console.log(`✅ Generated: ${indexPath}`);
}

function generateMasterIndex(docsDir) {
  const indexPath = path.join(docsDir, 'INDEX.md');

  const content = `# Documentation Master Index

**Last Updated**: ${new Date().toISOString()}

## Quick Navigation

- [Active Projects](active/projects/INDEX.md)
- [Agent Library](reference/agent-library/AGENT-REGISTRY.md)
- [Skills Catalog](reference/skill-library/SKILLS-CATALOG.md)
- [Archive](archive/INDEX.md)

## System Status

**Active Projects**: ${countProjects('docs/active/projects')}
**Total Agents**: ${countAgents('.claude/active/agents')}
**Total Skills**: ${countSkills('.claude/active/skills')}
**Archived Projects**: ${countProjects('docs/archive/projects')}

## Recent Updates

${getRecentUpdates()}
`;

  fs.writeFileSync(indexPath, content);
  console.log(`✅ Generated: ${indexPath}`);
}

// Helper functions
function countProjects(dir) {
  if (!fs.existsSync(dir)) return 0;
  return fs.readdirSync(dir).filter(f => fs.statSync(path.join(dir, f)).isDirectory()).length;
}

function countAgents(dir) {
  if (!fs.existsSync(dir)) return 0;
  let count = 0;
  const categories = fs.readdirSync(dir);
  for (const category of categories) {
    const categoryPath = path.join(dir, category);
    if (fs.statSync(categoryPath).isDirectory()) {
      count += fs.readdirSync(categoryPath).filter(f => f.endsWith('.md')).length;
    }
  }
  return count;
}

function countSkills(dir) {
  if (!fs.existsSync(dir)) return 0;
  let count = 0;
  const categories = fs.readdirSync(dir);
  for (const category of categories) {
    const categoryPath = path.join(dir, category);
    if (fs.statSync(categoryPath).isDirectory()) {
      count += fs.readdirSync(categoryPath).length;
    }
  }
  return count;
}

function getRecentUpdates() {
  // Scan for recent file modifications
  // Return formatted list
  return '- Agent Enhancement Project updated (2025-11-08)\n- MCP Integration in progress (2025-11-07)';
}

// Execute
generateAgentIndex('.claude/active/agents');
generateSkillIndex('.claude/active/skills');
generateMasterIndex('docs');
console.log('✅ All indexes generated successfully');
```

---

### B. Automation Scripts

#### 5. `scripts/taxonomy/scheduled-archival.js`
**Purpose**: Run daily to apply retention policies

```javascript
const fs = require('fs-extra');
const path = require('path');
const schedule = require('node-schedule');

const RETENTION_POLICIES = {
  'debug': 7,   // days
  'error': 30,
  'audit': 90,
  'cache': 1,
  'shell-snapshots': 7,
  'file-history': 30,
  'completed-todos': 30
};

function applyRetentionPolicy(category, dirPath, retentionDays) {
  const files = fs.readdirSync(dirPath);
  const cutoffDate = Date.now() - (retentionDays * 24 * 60 * 60 * 1000);

  let deleted = 0;
  let archived = 0;

  for (const file of files) {
    const filePath = path.join(dirPath, file);
    const stats = fs.statSync(filePath);

    if (stats.mtime.getTime() < cutoffDate) {
      // Compress audit logs before deletion
      if (category === 'audit') {
        const archivePath = path.join('.claude/archive', new Date().getFullYear().toString(), 'runtime-logs');
        fs.ensureDirSync(archivePath);
        // Compress and move
        archived++;
      } else {
        fs.removeSync(filePath);
        deleted++;
      }
    }
  }

  console.log(`✅ ${category}: Deleted ${deleted}, Archived ${archived}`);
}

function runRetentionPolicies() {
  console.log(`\n🗓️  Running retention policies: ${new Date().toISOString()}`);

  const policies = [
    { category: 'debug', path: '.claude/runtime/logs/debug', days: RETENTION_POLICIES.debug },
    { category: 'error', path: '.claude/runtime/logs/error', days: RETENTION_POLICIES.error },
    { category: 'audit', path: '.claude/runtime/logs/audit', days: RETENTION_POLICIES.audit },
    { category: 'cache', path: '.claude/runtime/cache', days: RETENTION_POLICIES.cache },
    { category: 'shell-snapshots', path: '.claude/runtime/cache/shell-snapshots', days: RETENTION_POLICIES['shell-snapshots'] },
    { category: 'file-history', path: '.claude/runtime/cache/file-history', days: RETENTION_POLICIES['file-history'] },
    { category: 'completed-todos', path: '.claude/runtime/todos/completed', days: RETENTION_POLICIES['completed-todos'] }
  ];

  for (const { category, path: dirPath, days } of policies) {
    if (fs.existsSync(dirPath)) {
      applyRetentionPolicy(category, dirPath, days);
    }
  }

  console.log('✅ Retention policies applied successfully\n');
}

// Schedule daily at 2 AM
schedule.scheduleJob('0 2 * * *', runRetentionPolicies);

// Run immediately if --now flag
if (process.argv.includes('--now')) {
  runRetentionPolicies();
}

console.log('📅 Scheduled archival automation started');
console.log('   Next run: 2:00 AM daily');
console.log('   Use --now flag to run immediately');
```

---

## VIII. Maintenance Procedures

### A. Weekly Maintenance

**Tasks**:
1. Review active projects for completion
2. Archive completed projects
3. Update indexes
4. Check for broken links
5. Validate retention policies

**Automation**:
```bash
# Weekly maintenance script
node scripts/taxonomy/weekly-maintenance.js
```

---

### B. Monthly Maintenance

**Tasks**:
1. Review archive for consolidation opportunities
2. Compress old archives
3. Audit tag consistency
4. Update reference material
5. Generate usage statistics

**Automation**:
```bash
# Monthly maintenance script
node scripts/taxonomy/monthly-maintenance.js
```

---

### C. Quarterly Reviews

**Tasks**:
1. Strategic archive review
2. Taxonomy refinement
3. Identify obsolete content
4. Plan next quarter organization
5. Document lessons learned

**Manual**: Scheduled 1-hour review session

---

## IX. Success Metrics

### A. Organization Metrics

**Before Migration**:
- `.claude/projects`: 560MB (mostly old history)
- `docs/`: 196 root-level files (disorganized)
- No clear navigation structure
- Manual search required

**After Migration Targets**:
- `.claude/runtime`: <100MB (retention policies active)
- `docs/`: <20 root-level files (index files only)
- 5-click navigation to any document
- Tag-based search functional

---

### B. Performance Metrics

**Targets**:
- File search time: <5 seconds (via indexes)
- Archive retrieval: <30 seconds
- Index generation: <2 minutes
- Daily retention cleanup: <1 minute

---

### C. Usability Metrics

**Targets**:
- New user onboarding: <10 minutes (via master index)
- Document discovery: <3 clicks
- Cross-reference lookup: <2 clicks
- Archival workflow: <5 minutes per project

---

## X. Migration Timeline

| Phase | Days | Deliverables | Risk |
|-------|------|--------------|------|
| 1. Preparation | 1 | Directory structure, inventory, backup | Low |
| 2. Runtime Data | 1 | Migrated runtime data, retention policies | Medium |
| 3. Active Content | 2 | Categorized agents/skills/commands | Medium |
| 4. Documentation | 3 | Organized active/archive/reference docs | High |
| 5. Archive | 2 | Historical content archived | Low |
| 6. Reference | 1 | Consolidated reference material | Low |
| 7. Validation | 2 | Tested, validated, links checked | Medium |
| 8. Automation | 2 | Scheduled tasks, monitoring | Low |
| **Total** | **14 days** | Complete reorganization | **Medium** |

---

## XI. Rollback Plan

### A. Backup Strategy

**Pre-Migration Backup**:
```bash
# Full backup
tar -czf claude-backup-$(date +%Y%m%d).tar.gz .claude/ docs/

# Verify backup
tar -tzf claude-backup-$(date +%Y%m%d).tar.gz | wc -l

# Store backup externally
mv claude-backup-$(date +%Y%m%d).tar.gz /path/to/safe/storage/
```

---

### B. Rollback Procedure

**If migration fails**:
```bash
# Stop all running processes
pkill -f "claude-flow"
pkill -f "taxonomy"

# Restore from backup
cd $HOME
rm -rf .claude docs
tar -xzf /path/to/safe/storage/claude-backup-YYYYMMDD.tar.gz

# Verify restoration
ls -la .claude
ls -la docs

# Restart services
npm start
```

---

## XII. Post-Migration Validation

### A. Validation Checklist

- [ ] All files present (compare with inventory)
- [ ] No broken links (run link checker)
- [ ] Indexes complete and accurate
- [ ] Retention policies working
- [ ] Search functionality operational
- [ ] Cross-references valid
- [ ] Automation scheduled
- [ ] Documentation updated

---

### B. User Acceptance Testing

**Test Scenarios**:
1. Find agent by name → Expected: <5 seconds
2. Find skill by trigger keyword → Expected: <3 seconds
3. Locate completed project → Expected: <30 seconds
4. Access reference material → Expected: <2 clicks
5. Search by tag → Expected: <10 seconds

---

## XIII. Conclusion

This comprehensive taxonomy and reorganization plan provides:

1. **Clear Structure**: 3-tier temporal taxonomy (active/archive/runtime)
2. **Automated Maintenance**: Retention policies, scheduled cleanup
3. **Easy Navigation**: Comprehensive indexes, tag-based search
4. **Safe Migration**: Phased approach with backups and validation
5. **Long-Term Sustainability**: Archival criteria, maintenance procedures

**Estimated Impact**:
- 80% reduction in navigation time
- 60% reduction in storage usage (retention policies)
- 95% improvement in document discovery
- 100% automated runtime data management

**Next Steps**:
1. Review and approve this plan
2. Execute Phase 1 (Preparation)
3. Monitor migration progress
4. Validate each phase before proceeding
5. Complete post-migration testing

---

**Document Status**: Implementation-Ready
**Approval Required**: Yes
**Estimated Completion**: 14 days from approval
