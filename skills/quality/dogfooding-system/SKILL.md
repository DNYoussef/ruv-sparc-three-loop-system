---
name: dogfooding-system
description: Self-improvement system enabling Claude Code to detect code quality violations,
  retrieve past fix patterns from memory, and orchestrate automated improvement cycles.
  Uses Connascence Analyzer for 7+ violation types, Memory-MCP for pattern storage
  with WHO/WHEN/PROJECT/WHY tagging, and sandbox testing with automated rollback.
  Perfect for continuous quality improvement and self-healing codebases.
version: 1.0.0
category: quality
tags:
- quality
- testing
- validation
author: ruv
---

# Dogfooding System - Self-Improvement Architecture

A comprehensive 3-phase self-improvement system that enables Claude Code to automatically improve itself and connected MCP servers through quality detection, pattern retrieval, and safe automated fixes.

## Overview

The Dogfooding System orchestrates three integrated phases:

1. **Quality Detection** - Run Connascence Analysis, detect 7+ violation types, store in Memory-MCP
2. **Pattern Retrieval** - Query Memory-MCP via vector search, rank patterns by similarity + success rate
3. **Continuous Improvement** - Full cycle orchestration with sandbox testing and automated rollback

**Key Components**:
- **Claude Code**: 106+ skills, 131+ agents for orchestration
- **Connascence Analyzer MCP**: 7+ violation types (God Objects, Parameter Bombs, Deep Nesting, etc.)
- **Memory-MCP Triple System**: Persistent cross-session memory with semantic search and metadata tagging

## When to Use This Skill

Activate this skill when:
- Need to analyze code quality across projects systematically
- Want to find and apply past fix patterns for similar violations
- Running automated improvement cycles on MCP servers or Claude Code itself
- Implementing self-healing workflows with safety guarantees
- Building institutional knowledge about code quality patterns
- Establishing continuous improvement processes

This skill is particularly valuable when:
- Working with large codebases requiring systematic quality improvement
- Building self-improving systems that learn from past fixes
- Implementing production-ready dogfooding workflows with safety rails
- Creating automated code quality pipelines with verification

## Three-Phase Architecture

### Phase 1: Quality Detection (30-60 seconds)

Run Connascence Analysis to detect violations and store findings in Memory-MCP.

**Workflow**:
1. Run Connascence Analyzer on target project
2. Parse 7+ violation types with severity levels
3. Store each violation in Memory-MCP with WHO/WHEN/PROJECT/WHY tags
4. Update Grafana dashboard with metrics
5. Generate summary report

**Violations Detected**:
- God Objects (>15 methods)
- Parameter Bombs/CoP (>6 params, NASA limit)
- Cyclomatic Complexity (>10)
- Deep Nesting (>4 levels, NASA limit)
- Long Functions (>50 lines)
- Magic Literals/CoM (hardcoded values)
- Duplicate Code Blocks

**Execution**:
```bash
# Run quality detection for single project
.\resources\scripts\run-quality-detection.bat memory-mcp

# Run for all projects
.\resources\scripts\run-quality-detection.bat all
```

**Agents**: `code-analyzer`, `reviewer`

**Outputs**:
- JSON: `metrics/dogfooding/<project>_<timestamp>.json`
- Summary: `metrics/dogfooding/summary_<timestamp>.txt`
- Memory-MCP: Tagged violation records
- Dashboard: Updated metrics

### Phase 2: Pattern Retrieval (10-30 seconds)

Query Memory-MCP for similar past fixes using vector search, rank by relevance, optionally apply.

**Workflow**:
1. Receive violation from Phase 1 (or manual query)
2. Generate 384-dimensional embedding via `all-MiniLM-L6-v2`
3. Query ChromaDB with HNSW indexing for top-5 similar patterns
4. Rank by: similarity (40%) + success_rate (30%) + context_match (20%) + recency (10%)
5. Select best transformation strategy
6. (Optional) Apply AST transformation with sandbox testing

**Vector Search**:
- Model: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- Backend: ChromaDB with HNSW indexing
- Similarity: Cosine distance (1 - distance = similarity score)
- Filters: intent, project, agent_category metadata

**Transformation Strategies**:
1. **Delegation Pattern** - God Object → separate classes
2. **Config Object Pattern** - Parameter Bomb → object param
3. **Early Return Pattern** - Deep Nesting → guard clauses
4. **Extract Method Pattern** - Long Function → smaller functions
5. **Named Constant Pattern** - Magic Literal → const
6. **Extract Function Pattern** - Duplicate Code → DRY

**Execution**:
```bash
# Query only (no application)
.\resources\scripts\run-pattern-retrieval.bat "God Object with 26 methods"

# Query + apply best pattern
.\resources\scripts\run-pattern-retrieval.bat "Parameter Bomb 10 params" --apply
```

**Agents**: `code-analyzer`, `coder`, `reviewer`

**Outputs**:
- JSON: `retrievals/query-<timestamp>.json` (vector search results)
- Best pattern: `retrievals/best-pattern-<timestamp>.json`
- (Optional) Applied transformation with git commit

### Phase 3: Continuous Improvement (60-120 seconds)

Full cycle orchestration combining Quality Detection + Pattern Retrieval + Safe Application.

**Workflow**:
1. **Quality Detection** (30-60s) - Detect violations
2. **Pattern Retrieval** (10-30s) - Find best fixes
3. **Safe Application** (20-40s) - Apply in sandbox first
4. **Verification** (15s) - Re-run analysis, compare metrics
5. **Summary & Metrics** (10-20s) - Generate reports, update dashboard

**Safety Checks** (MANDATORY):
1. ✅ Sandbox testing REQUIRED before production
2. ✅ Automated rollback via git stash
3. ✅ Progressive application (one fix at a time)
4. ✅ Test coverage ≥70% required
5. ✅ CI/CD gate must pass before merge

**Execution**:
```bash
# Single cycle with safety checks
.\resources\scripts\run-continuous-improvement.bat memory-mcp

# Dry-run (no fixes applied)
.\resources\scripts\run-continuous-improvement.bat memory-mcp --dry-run

# Full cycle all projects (round-robin)
.\resources\scripts\run-continuous-improvement.bat all
```

**Agents**: `hierarchical-coordinator`, `code-analyzer`, `coder`, `reviewer`

**Outputs**:
- Cycle summary: `cycle-summaries/cycle-<id>.txt`
- Archive: `archive/<cycle_id>/` (all artifacts)
- Dashboard: Updated Grafana metrics
- Memory-MCP: Cycle summary + all fixes stored
- Git commits: Fixes with safety metadata

## Memory-MCP Tagging Protocol

ALL Memory-MCP writes use automatic metadata tagging:

**WHO**: Agent name, category, capabilities
**WHEN**: ISO timestamp, Unix timestamp, readable format
**PROJECT**: connascence-analyzer, memory-mcp-triple-system, claude-flow, etc.
**WHY**: Intent (implementation, bugfix, refactor, testing, documentation, analysis, planning, research)

**Implementation**:
```javascript
const { taggedMemoryStore } = require('./hooks/12fa/memory-mcp-tagging-protocol.js');

// Auto-tagged memory write
const tagged = taggedMemoryStore('code-analyzer', 'Detected God Object with 26 methods', {
  violation_type: 'god-object',
  severity: 'high',
  project: 'memory-mcp'
});
// Automatically includes: agent metadata, timestamps, project, intent
```

## Safety Rules (CRITICAL)

Documentation: `C:\Users\17175\docs\DOGFOODING-SAFETY-RULES.md`

### 1. Sandbox Testing REQUIRED
```bash
mkdir C:\Users\17175\tmp\dogfood-sandbox
xcopy /E /I /Q <project> C:\Users\17175\tmp\dogfood-sandbox
cd C:\Users\17175\tmp\dogfood-sandbox && npm test
# If pass → apply to production
# If fail → reject fix
```

### 2. Automated Rollback
```bash
git stash push -u -m "backup-<timestamp>"
<apply-fix>
npm test || git stash pop  # Rollback on failure
```

### 3. Progressive Application
- Fix ONE violation at a time
- Test after each fix
- Commit after each successful fix
- Never batch fixes without testing

### 4. Test Coverage Requirement
- ONLY apply fixes to code with ≥70% test coverage
- Add tests FIRST if coverage is insufficient

### 5. CI/CD Gate
- ALL fixes must pass CI/CD before merge
- Automated safety checks in `.github/workflows/dogfooding-safety.yml`

## Metrics Tracked

### Phase 1 (Quality Detection)
- Analysis Duration (target: <30s)
- Violations Detected (by type)
- Files Analyzed
- Storage Success Rate (target: 100%)
- Dashboard Update Status

### Phase 2 (Pattern Retrieval)
- Query Time (target: <1000ms)
- Results Found (target: ≥3)
- Top Similarity Score (target: ≥0.70)
- Application Success Rate (target: ≥95%)
- Improvement Magnitude

### Phase 3 (Continuous Improvement)
- Cycle Duration (target: <120s)
- Violations Fixed (target: ≥3 per cycle)
- Success Rate (target: ≥95%)
- Improvement Velocity (target: ≥5 violations/day)
- Pattern Retrieval Quality (target: ≥0.75 avg similarity)
- Sandbox Testing Pass Rate (target: 100%)
- Production Rollback Rate (target: ≤5%)

## Quick Start

### 1. Run Quality Detection Only
```bash
.\resources\scripts\run-quality-detection.bat memory-mcp
```

### 2. Query Past Fixes
```bash
.\resources\scripts\run-pattern-retrieval.bat "God Object with 26 methods"
```

### 3. Run Full Improvement Cycle
```bash
.\resources\scripts\run-continuous-improvement.bat memory-mcp
```

### 4. Schedule Automated Cycles
```bash
# Windows Task Scheduler (daily at 12:00 UTC)
schtasks /create /tn "Dogfooding-Cycle" \
  /tr "C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\skills\dogfooding-system\resources\scripts\run-continuous-improvement.bat all" \
  /sc daily /st 12:00
```

## Bundled Resources

### Scripts
- `run-quality-detection.bat` - Execute Phase 1 (Quality Detection)
- `run-pattern-retrieval.bat` - Execute Phase 2 (Pattern Retrieval)
- `run-continuous-improvement.bat` - Execute Phase 3 (Full Cycle)
- `generate-cycle-summary.js` - Generate cycle summary reports

### Templates
- `violation-report.md` - Violation report template with metadata
- `fix-pattern.json` - Fix pattern schema for Memory-MCP storage
- `cycle-summary.md` - Cycle summary template with metrics

### Tests
- `test-quality-detection.js` - Verify Phase 1 workflow
- `test-pattern-retrieval.js` - Verify Phase 2 vector search
- `test-continuous-improvement.js` - Verify Phase 3 full cycle

## Troubleshooting

### Issue: VectorIndexer has no attribute 'collection'
**Fix**: Already patched in Memory-MCP
```python
def __init__(self, ...):
    self.client = chromadb.PersistentClient(path=persist_directory)
    self.create_collection()  # <-- Added this line
```

### Issue: No patterns found in Memory-MCP
**Solution**: Run Phase 1 (Quality Detection) first to populate Memory-MCP with violations and fixes

### Issue: Sandbox tests pass but production fails
**Solution**: Enhanced sandbox testing to better replicate production environment
- Check for environment-specific dependencies
- Verify test coverage is sufficient
- Update sandbox creation to match production more closely

## Integration with Other Skills

**Upstream Skills** (that feed into dogfooding):
- `functionality-audit` - Sandbox testing + debugging
- `theater-detection-audit` - Byzantine consensus verification
- `production-readiness` - Complete deployment checklist

**Downstream Skills** (that use dogfooding):
- `cicd-intelligent-recovery` - Automated failure recovery
- `performance-analysis` - Performance optimization
- `code-review-assistant` - Multi-agent swarm review

## File Structure

```
dogfooding-system/
├── skill.md (this file)
├── INDEX.md (comprehensive documentation)
├── resources/
│   ├── scripts/
│   │   ├── run-quality-detection.bat
│   │   ├── run-pattern-retrieval.bat
│   │   ├── run-continuous-improvement.bat
│   │   └── generate-cycle-summary.js
│   └── templates/
│       ├── violation-report.md
│       ├── fix-pattern.json
│       └── cycle-summary.md
├── tests/
│   ├── test-quality-detection.js
│   ├── test-pattern-retrieval.js
│   └── test-continuous-improvement.js
└── graphviz/
    └── dogfooding-system-process.dot
```

## Success Metrics

Since implementation:
- ✅ VectorIndexer bug fixed (collection attribute initialization)
- ✅ 27 Unicode violations fixed in Connascence Analyzer
- ✅ 7 import issues fixed in Memory-MCP
- ✅ 45 violations detected in Memory-MCP codebase
- ✅ 46 fixes stored in Memory-MCP with proper metadata
- ✅ Vector search working with 0.82+ average similarity
- ✅ WHO/WHEN/PROJECT/WHY tagging protocol implemented
- ✅ Safety rules documented and enforced
- ✅ 3 SOP skills created with full agent assignments

**Status**: ✅ PRODUCTION READY
**Version**: 1.0 (Gold Tier)
**Skills**: 3 SOPs (Quality Detection, Pattern Retrieval, Continuous Improvement)
**Agents**: hierarchical-coordinator, code-analyzer, coder, reviewer
**MCP Tools**: connascence-analyzer, memory-mcp, claude-flow
**Safety**: Sandbox testing + automated rollback + verification