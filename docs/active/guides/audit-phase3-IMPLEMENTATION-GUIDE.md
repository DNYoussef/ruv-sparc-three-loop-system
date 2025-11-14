# IMPLEMENTATION GUIDE: Fixing sop-dogfooding-continuous-improvement

**Purpose**: Step-by-step instructions to fix all P1 critical issues  
**Estimated Time**: 4-5 hours  
**Skill**: sop-dogfooding-continuous-improvement  
**Status**: Ready to implement

---

## TASK 1: Metadata Reconciliation (30 minutes)

### What to fix
Frontmatter is incomplete - missing 50% of scripts and tool details

### Current state (Lines 1-10)
```yaml
---
name: sop-dogfooding-continuous-improvement
description: 3-part dogfooding workflow Phase 3 - Full cycle orchestration combining Quality Detection + Pattern Retrieval + Application with automated metrics tracking. 60-120 seconds execution time.
agents: hierarchical-coordinator, code-analyzer, coder, reviewer
mcp_tools: connascence-analyzer, memory-mcp, claude-flow
scripts: dogfood-continuous-improvement.bat, generate-cycle-summary.js, update-dashboard.js
---
```

### Implementation steps

**Step 1**: Open the SKILL.md file
```bash
C:\Users\17175\skills\sop-dogfooding-continuous-improvement\SKILL.md
```

**Step 2**: Replace frontmatter (Lines 1-10) with complete metadata

```yaml
---
name: sop-dogfooding-continuous-improvement
description: >
  Phase 3 of 3-part dogfooding system: Orchestrates complete automated improvement cycles 
  combining quality detection, pattern retrieval, safe application (with sandbox testing), 
  verification, summary generation, and metrics tracking. 
  Execution time: 60-120 seconds per cycle.

agents:
  - hierarchical-coordinator: Orchestration, initialization, summary generation, dashboard updates, cleanup
  - code-analyzer: Quality detection (Phase 2), pattern retrieval (Phase 3)
  - coder: Safe fix application with sandbox testing (Phase 4)
  - reviewer: Verification and regression detection (Phase 5)

mcp_tools:
  - connascence-analyzer:
      health_check: Pre-flight verification (Phase 1)
      analyze_workspace: Quality detection and re-analysis (Phases 2, 5)
  - memory-mcp:
      vector_search: Pattern retrieval (Phase 3)
      memory_store: Cycle metadata and results storage (Phases 1-8)
  - claude-flow:
      hooks: Agent coordination signaling (all phases)
      task_orchestrate: Next-cycle scheduling (Phase 7)

scripts:
  - dogfood-continuous-improvement.bat: Main entry point
  - dogfood-quality-check.bat: Phase 2 quality detection
  - query-memory-mcp.js: Phase 3 pattern retrieval
  - apply-fix-pattern.js: Phase 4 fix application
  - generate-cycle-summary.js: Phase 6 summary generation
  - update-dashboard.js: Phase 7 dashboard updates

trigger_keywords:
  - continuous improvement
  - dogfooding cycle
  - automated code quality
  - violation reduction
  - pattern-based fixes

execution_time: 60-120 seconds per cycle
dependencies:
  - sop-dogfooding-quality-detection
  - sop-dogfooding-pattern-retrieval
  - DOGFOODING-SAFETY-RULES.md
---
```

**Step 3**: Verify changes
- [ ] All 6 scripts listed
- [ ] All 3 MCP tools listed with specific functions
- [ ] All 4 agents listed with descriptions
- [ ] Trigger keywords added
- [ ] Dependencies documented

**Time estimate**: 15 minutes

---

## TASK 2: Add Data Flow Section (45 minutes)

### What to fix
No documentation of how phase outputs become next phase inputs

### Where to insert
After System Architecture diagram (after line 39)

### Implementation steps

**Step 1**: Insert new section before "## Phase 1: Initialize Cycle"

Add this content:

```markdown
## Data Flow & Integration

This section shows how data flows between phases and how agents communicate.

### Phase-to-Phase Data Flow

```
INPUT: Cycle metadata from Phase 1
  ↓
PHASE 2: Quality Detection
  Input: target_project (string)
  Output: violations.json {total: X, critical: Y, violations: [...]}
  Storage: Memory-MCP key "dogfooding/cycle/<cycle_id>/violations"
  ↓

PHASE 3: Pattern Retrieval
  Input: violations.json from Phase 2
  Query: For each violation, search Memory-MCP for similar patterns
  Output: best-pattern-<id>.json {pattern_name, similarity_score, fallback_used}
  Storage: Memory-MCP key "dogfooding/cycle/<cycle_id>/patterns"
  ↓

PHASE 4: Safe Application
  Input: best-pattern-*.json files from Phase 3
  Process: For each pattern, sandbox test → production test → commit
  Output: application-results.json {applied: true/false, sandbox_pass: T/F, prod_pass: T/F}
  Storage: Memory-MCP + Git commits
  ↓

PHASE 5: Verification
  Input: application-results.json from Phase 4
  Process: Re-analyze project, compare before/after
  Output: verification-report.json {violations_before, violations_after, improvement%, regressions}
  Decision: If regressions → ROLLBACK, else → continue
  ↓

PHASE 6: Summary Generation
  Input: All previous outputs (violations, patterns, applications, verification)
  Process: Aggregate metrics, format human-readable summary
  Output: cycle-<id>.txt + cycle-<id>.json
  Storage: File system + Memory-MCP
  ↓

PHASE 7: Dashboard Updates
  Input: cycle-<id>.json from Phase 6
  Process: Update SQLite, refresh Grafana, trigger hooks
  Output: Dashboard metrics updated, next cycle scheduled
  ↓

PHASE 8: Cleanup & Archive
  Input: cycle_id from earlier phases
  Process: Archive artifacts, remove temp files, log completion
  Output: Artifacts in archive/, temp files deleted
```

### Agent Communication Pattern

Each agent communicates via:
1. **Memory-MCP**: Persistent cross-phase storage
   - Write: `memory_store("key", {data})` with metadata
   - Read: `vector_search("query")` for patterns
   
2. **Git**: Version control for applied fixes
   - Commits include cycle metadata in message
   - Stash/pop for sandbox safety

3. **Files**: JSON/text output files at known paths
   - Violations: `metrics/dogfooding/<project>_<timestamp>.json`
   - Patterns: `metrics/dogfooding/retrievals/best-pattern-*.json`
   - Results: `metrics/dogfooding/cycle-summaries/cycle-<id>.txt`

### Data Format Examples

**Violations JSON** (Phase 2 output):
```json
{
  "project": "memory-mcp-triple-system",
  "timestamp": "2025-11-02T12:00:00Z",
  "total_violations": 45,
  "by_severity": {
    "CRITICAL": 8,
    "HIGH": 12,
    "MEDIUM": 25
  },
  "violations": [
    {
      "id": "PARAM_BOMB_a3f2_142",
      "type": "PARAMETER_BOMB",
      "file": "src/vectorizer.js",
      "line": 142,
      "severity": "CRITICAL",
      "message": "Function has 14 parameters (NASA limit: 6)"
    }
  ]
}
```

**Best Pattern JSON** (Phase 3 output):
```json
{
  "violation_id": "PARAM_BOMB_a3f2_142",
  "violation_type": "PARAMETER_BOMB",
  "pattern_selected": "extract-function-by-group",
  "similarity_score": 0.82,
  "fallback_used": false,
  "confidence": "high",
  "retrieved_patterns": [
    {"rank": 1, "name": "extract-function-by-group", "score": 0.82},
    {"rank": 2, "name": "reduce-params-via-config", "score": 0.71},
    {"rank": 3, "name": "use-options-object", "score": 0.65}
  ]
}
```

**Application Results JSON** (Phase 4 output):
```json
{
  "pattern_id": "extract-function-by-group",
  "violation_id": "PARAM_BOMB_a3f2_142",
  "target_file": "src/vectorizer.js",
  "applied": true,
  "sandbox_tested": true,
  "sandbox_passed": true,
  "production_tested": true,
  "production_passed": true,
  "git_commit": "a1b2c3d - dogfooding: Applied extract-function-by-group to vectorizer.js",
  "violations_before": 45,
  "violations_after": 44
}
```
```

**Step 2**: Verify insertion
- [ ] New section placed after System Architecture (line ~40)
- [ ] Before "## Phase 1" section
- [ ] All example JSONs properly formatted

**Step 3**: Update Phase descriptions
For each phase, add this at the top:

```markdown
**Data Flow**:
- Input: [description of data this phase receives]
- Output: [description of data this phase produces]
- Storage: [where data is stored for next phase]
```

**Time estimate**: 30 minutes

---

## TASK 3: Refactor Long Prompts (2 hours)

### What to fix
Phase 3 and 4 prompts are 70-110+ lines, making them hard to understand/execute

### PART A: Phase 3 Refactor

**Step 1**: Find current Phase 3 section (Line ~207)

Current structure:
```
## Phase 3: Execute Pattern Retrieval (10-30 sec)

**Agent**: `code-analyzer`

**Prompt**:
```javascript
await Task("Pattern Retrieval Executor", `
[110+ LINES OF DETAILED INSTRUCTIONS]
`, "code-analyzer");
```
```

**Step 2**: Replace with condensed version

```markdown
## Phase 3: Execute Pattern Retrieval (10-30 sec)

**Agent**: `code-analyzer`

**Responsibility**: For each violation found in Phase 2, retrieve best-matching fix patterns from Memory-MCP.

**Timeline**: 10-30 seconds per cycle

**Process**:
1. Read violations from Phase 2 output file
2. For each CRITICAL violation (prioritize NASA compliance):
   - Formulate semantic search query
   - Execute: `node query-memory-mcp.js --query "<query>" --limit 5`
   - Rank results by: similarity(0.4) + success_rate(0.3) + context(0.2) + recency(0.1)
   - If similarity ≥ 0.70 → Select, else → Use fallback strategy
3. Store best patterns in `metrics/dogfooding/retrievals/best-pattern-<id>.json`
4. Aggregate results: patterns_found, patterns_missing, avg_similarity

**Delegation**: Delegates to `sop-dogfooding-pattern-retrieval` skill for detailed implementation

**Implementation Details**: See [Phase 3 Implementation Guide](./phase3-pattern-retrieval-detailed.md)

**Success Criteria**:
- ✅ All critical violations have patterns or fallback strategies
- ✅ Pattern files generated for Phase 4
- ✅ Average similarity ≥ 0.70 for selected patterns
- ✅ Fallback strategies documented for low-similarity cases
```

**Step 3**: Create separate implementation file

Create new file: `C:\Users\17175\docs\phase3-pattern-retrieval-detailed.md`

Content:
```markdown
# Phase 3: Pattern Retrieval - Detailed Implementation

## Overview
Retrieves best-matching fix patterns from Memory-MCP for violations detected in Phase 2.

## Full Process

### Input Files
- `C:\Users\17175\metrics\dogfooding\<project>_<timestamp>.json` (violations from Phase 2)

### Step-by-Step

#### STEP 1: Initialize Phase (2 sec)
- Read violations JSON file
- Load MCP connection to Memory-MCP
- Initialize ranking algorithm weights

#### STEP 2: Process Each Critical Violation (5-20 sec)
```bash
For each violation with severity = "CRITICAL":
  1. Formulate search query
     Example: "Fix Parameter Bomb with 14 parameters to meet NASA limit of 6"
  
  2. Execute Memory-MCP vector search
     Command: node C:\Users\17175\scripts\query-memory-mcp.js \
       --query "<query>" \
       --limit 5 \
       --project <target_project>
  
  3. Receive ranked results (top 5)
     Results: [{pattern_name, similarity_score, success_rate, context_match, recency, ...}]
  
  4. Rank by weighted formula
     rank_score = (similarity * 0.4) + (success_rate * 0.3) + \
                  (context_match * 0.2) + (recency * 0.1)
  
  5. Evaluate: Is rank_score ≥ 0.70?
     YES → Select pattern, store in best-pattern-<id>.json
     NO → Apply fallback strategy (see below)
```

#### STEP 3: Fallback Strategies (for similarity < 0.70)

**Fallback Order**:
1. **Fallback A1**: Use next-best pattern (similarity 0.65-0.70)
   - Lower confidence but still useful
   - Mark as "fallback_used: true"

2. **Fallback A2**: Apply generic refactoring template
   - Example: "Extract function to reduce parameter count"
   - Violation type → Template mapping
   - Mark with: "fallback_strategy": "generic_template"

3. **Fallback A3**: Mark for manual review
   - No pattern or template available
   - Requires human judgment
   - Mark with: "action": "manual_review"

#### STEP 4: Store Best Pattern or Fallback

Output file: `C:\Users\17175\metrics\dogfooding\retrievals\best-pattern-<violation_id>.json`

```json
{
  "violation_id": "PARAM_BOMB_a3f2_142",
  "violation_type": "PARAMETER_BOMB",
  "file": "src/vectorizer.js",
  "line": 142,
  "search_query": "Fix Parameter Bomb with 14 parameters",
  "search_results": [
    {"rank": 1, "name": "extract-function-by-group", "score": 0.82, "success_rate": 0.92},
    {"rank": 2, "name": "reduce-params-via-config", "score": 0.71, "success_rate": 0.78},
    {"rank": 3, "name": "use-options-object", "score": 0.65, "success_rate": 0.68}
  ],
  "pattern_selected": "extract-function-by-group",
  "similarity_score": 0.82,
  "success_rate": 0.92,
  "rank_score": 0.794,  // (0.82*0.4) + (0.92*0.3) + (0.85*0.2) + (0.9*0.1)
  "fallback_used": false,
  "confidence": "high",
  "selected_at": "2025-11-02T12:00:15Z"
}
```

#### STEP 5: Aggregate Results

Store summary in Memory-MCP:
```json
{
  "phase": 3,
  "cycle_id": "cycle-20251102120000",
  "total_violations": 45,
  "violations_with_patterns": 42,
  "violations_with_fallback": 2,
  "violations_without_strategy": 1,
  "avg_similarity_score": 0.82,
  "avg_success_rate": 0.85,
  "high_confidence_patterns": 38,
  "medium_confidence_patterns": 4,
  "low_confidence_patterns": 0,
  "manual_review_required": 1,
  "status": "COMPLETE"
}
```

### Error Handling

If Memory-MCP unavailable:
```
1. Check MCP health: npx claude-flow hooks mcp-status --tool memory-mcp
2. If error persists:
   - Use local pattern cache if available
   - Or skip to Phase 5 (verification without fixes)
   - Alert user
```

If search returns no results:
```
1. Apply Fallback A3 (manual review)
2. Store in metrics with action: "manual_review"
3. Continue with next violation
```

### Success Criteria

- ✅ All violations have patterns or fallback strategies
- ✅ Average similarity ≥ 0.70 for selected patterns
- ✅ Best patterns stored in correct format
- ✅ Summary metrics stored in Memory-MCP
- ✅ Phase 4 ready to proceed
```

**Step 4**: Verify
- [ ] Phase 3 section condensed to 10-15 lines
- [ ] Separate implementation file created
- [ ] Implementation file has all 70+ lines of detail

### PART B: Phase 4 Refactor

**Step 1**: Find current Phase 4 section (Line ~260)

**Step 2**: Replace with condensed version

```markdown
## Phase 4: Safe Application with Sandbox Testing (20-40 sec)

**Agent**: `coder`

**Responsibility**: Apply selected fix patterns with mandatory sandbox testing before production.

**Timeline**: 20-40 seconds per cycle

**Safety Gate**: SANDBOX TESTING REQUIRED - never apply to production without testing first

**Process**:
1. For each selected pattern (up to 5 per cycle):
   - Create sandbox: `mkdir C:\Users\17175\tmp\dogfood-sandbox-<id>`
   - Copy project: `xcopy /E /I <project> <sandbox>`
   - Apply fix in sandbox: `node apply-fix-pattern.js --input <pattern> --sandbox <sandbox>`
   - Test in sandbox: `cd <sandbox> && npm test`
   
2. Decision: Sandbox results?
   - PASS → Apply to production (step 3)
   - FAIL → Reject fix, mark pattern as failed, continue to next
   
3. Production application (for sandbox-passing fixes):
   - Backup: `git stash push -u -m "backup-<id>"`
   - Apply: `<apply-fix-commands>`
   - Test: `npm test`
   - Commit: `git commit -m "dogfooding: <pattern> - <improvement>"`
   
4. Decision: Production results?
   - PASS → Success, move to next pattern
   - FAIL → Rollback: `git reset --hard HEAD && git stash pop`

5. Cleanup: Remove sandbox directories

**Delegation**: Executes fixes found by Phase 3 patterns

**Implementation Details**: See [Phase 4 Implementation Guide](./phase4-safe-application-detailed.md)

**Success Criteria**:
- ✅ All sandbox tests pass before production
- ✅ All production tests still passing
- ✅ No broken code merged
- ✅ All failures automatically rolled back
- ✅ Success/failure metrics tracked
```

**Step 3**: Create separate implementation file

Create: `C:\Users\17175\docs\phase4-safe-application-detailed.md`

(Similar structure to Phase 3 - detailed steps, code blocks, error handling, success criteria)

**Step 4**: Verify
- [ ] Phase 4 section condensed to 10-15 lines
- [ ] Separate implementation file created
- [ ] All 90+ lines of detail preserved in separate file

**Time estimate**: 1.5 hours

---

## TASK 4: Add Resource Verification & Schemas (1 hour)

### What to fix
No SQLite schema documentation, MCP configuration unclear, script existence not verified

### Implementation steps

**Step 1**: Create database schema documentation

Create file: `C:\Users\17175\docs\dogfooding-database-schema.md`

Content:
```markdown
# Dogfooding Database Schema

## SQLite Database: dogfooding.db

Used by Phase 7 to track metrics and schedule cycles.

### Table: cycles

```sql
CREATE TABLE IF NOT EXISTS cycles (
  cycle_id TEXT PRIMARY KEY,
  target_project TEXT NOT NULL,
  started_at TIMESTAMP NOT NULL,
  completed_at TIMESTAMP NOT NULL,
  duration_seconds INTEGER NOT NULL,
  violations_before INTEGER NOT NULL,
  violations_after INTEGER NOT NULL,
  critical_before INTEGER,
  critical_after INTEGER,
  fixes_attempted INTEGER NOT NULL,
  fixes_successful INTEGER NOT NULL,
  fixes_failed INTEGER NOT NULL,
  sandbox_passes INTEGER NOT NULL,
  sandbox_fails INTEGER NOT NULL,
  success_rate REAL NOT NULL,
  improvement_percent REAL,
  regressions_found BOOLEAN DEFAULT FALSE,
  next_cycle_scheduled TIMESTAMP
);
```

### Table: patterns

```sql
CREATE TABLE IF NOT EXISTS patterns (
  pattern_id TEXT PRIMARY KEY,
  pattern_name TEXT NOT NULL,
  violation_type TEXT NOT NULL,
  times_retrieved INTEGER DEFAULT 0,
  times_applied INTEGER DEFAULT 0,
  times_successful INTEGER DEFAULT 0,
  success_rate REAL DEFAULT 0.0,
  avg_similarity_score REAL,
  last_used TIMESTAMP,
  created_at TIMESTAMP
);
```

### Table: violations

```sql
CREATE TABLE IF NOT EXISTS violations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cycle_id TEXT NOT NULL,
  violation_type TEXT NOT NULL,
  file_path TEXT NOT NULL,
  line_number INTEGER,
  severity TEXT,
  fixed_by_pattern TEXT,
  created_at TIMESTAMP,
  fixed_at TIMESTAMP,
  FOREIGN KEY (cycle_id) REFERENCES cycles(cycle_id)
);
```

### Initialization

```bash
sqlite3 C:\Users\17175\metrics\dogfooding\dogfooding.db < dogfooding-schema.sql
```
```

**Step 2**: Add MCP configuration guide section to SKILL.md

Insert after "## Phase 1: Initialize Cycle":

```markdown
## MCP Tool Configuration & Health Checks

### Connascence Analyzer

**Health Check**:
```bash
npx claude-flow hooks mcp-status --tool connascence-analyzer
# Expected output: HEALTHY
```

**Configuration**:
- Endpoint: `http://localhost:3000`
- Timeout: 30000ms
- Violation thresholds:
  - God object: >15 methods
  - Parameter bomb: >6 params (NASA limit)
  - Deep nesting: >4 levels (NASA limit)

### Memory-MCP

**Health Check**:
```bash
npx claude-flow hooks mcp-status --tool memory-mcp
# Expected output: HEALTHY and ChromaDB accessible
```

**Configuration**:
- Backend: ChromaDB
- Embedding: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- Persistence: `C:\Users\17175\.memory-mcp\`

### Claude Flow

**Health Check**:
```bash
npx claude-flow hooks mcp-status --tool claude-flow
# Expected output: HEALTHY
```

**Configuration**:
- Hooks available: pre-task, post-task, post-edit, session-end
- Timeout: 300000ms (5 minutes per phase)
```

**Step 3**: Add pre-flight validation section

Insert new section before "## Phase 1":

```markdown
## Pre-Flight Validation Checklist

Run this BEFORE executing any dogfooding cycle:

### System Check
```bash
# Verify Node.js
node --version
# Expected: ≥14.0.0

# Verify npm
npm --version
# Expected: ≥6.0.0

# Verify Git
git --version

# Verify Python (for connascence)
python --version
# Expected: ≥3.8
```

### Disk Space
```bash
wmic logicaldisk get size,freespace
# Need: ≥5GB free for artifacts and sandboxes
```

### MCP Health Checks
```bash
npx claude-flow hooks mcp-status --tool connascence-analyzer
# Expected: HEALTHY

npx claude-flow hooks mcp-status --tool memory-mcp
# Expected: HEALTHY

npx claude-flow hooks mcp-status --tool claude-flow
# Expected: HEALTHY
```

### Directory Verification
```bash
# Create required directories if missing
mkdir C:\Users\17175\metrics\dogfooding
mkdir C:\Users\17175\metrics\dogfooding\archive
mkdir C:\Users\17175\scripts
mkdir C:\Users\17175\tmp
```

### Script Existence
```bash
# Verify all scripts exist
dir C:\Users\17175\scripts\dogfood-*.bat
dir C:\Users\17175\scripts\*-memory-mcp.js
dir C:\Users\17175\scripts\apply-fix-pattern.js
dir C:\Users\17175\scripts\generate-cycle-summary.js
dir C:\Users\17175\scripts\update-dashboard.js
# All should list files with no errors
```

### Test Framework Verification
```bash
# Verify target project has working tests
cd C:\Users\17175\Desktop\memory-mcp-triple-system
npm test
# Should run (may fail tests, but must be runnable)
```

### Dry-Run Test
```bash
# Execute one cycle without applying fixes
C:\Users\17175\scripts\dogfood-continuous-improvement.bat memory-mcp --dry-run
# Expected: Complete in 30-60 seconds, no changes to code
```

**All checks must pass before proceeding to actual cycle.**
```

**Time estimate**: 45 minutes

---

## TASK 5: Standardize Instruction Format (30 minutes)

### What to fix
Instructions mix JavaScript Task() syntax, bash commands, pseudo-code, and English narrative

### Implementation steps

**Step 1**: Create instruction template

Add new section to SKILL.md after "## Phase 8: Cleanup":

```markdown
## Instruction Format Standards

All phase instructions follow this template:

```markdown
### Phase N: [Phase Name]

**Agent**: `agent-name`

**Responsibility**: [1-2 sentence description]

**Timeline**: [X-Y seconds]

**Inputs**:
- Data from Phase N-1: [description]
- Query Memory-MCP: [if applicable]

**Process** (imperative, numbered steps):
1. [Action] - [what and why]
   - Sub-action A
   - Sub-action B
2. [Decision point]
   - If condition A → [path X]
   - If condition B → [path Y]

**Outputs**:
- File: [path, format]
- Memory-MCP: [key, format]
- Git: [commits, branches]

**Success Criteria** (all must pass):
- ✅ [Criterion 1]
- ✅ [Criterion 2]

**Failure Handling**:
- If [failure] → [recovery]
```

### Execution Patterns

**Pattern 1: Script Execution**
```
Execute script <name>:
  Command: <full command with parameters>
  Input file: <path>
  Output file: <path>
  Expected: <what happens on success>
```

**Pattern 2: Conditional Logic**
```
Decision: <What are we deciding?>
  If condition A → Take path X
  If condition B → Take path Y
  Otherwise → Default action Z
```

**Pattern 3: Error Recovery**
```
If <error condition>:
  1. Immediate action
  2. Recovery step
  3. Retry or skip
  4. Log failure
```
```

**Step 2**: Apply template to all existing phases

For each phase section, ensure it:
- [ ] Uses imperative voice ("Execute", "Apply", "Verify", not "This phase")
- [ ] Numbers all steps (1, 2, 3...)
- [ ] Shows inputs clearly
- [ ] Shows outputs clearly
- [ ] Lists success criteria with checkmarks (✅)
- [ ] Documents failure handling

**Step 3**: Verify document consistency

Review all 8 phases:
- [ ] Phase 1: Follows template
- [ ] Phase 2: Follows template
- [ ] Phase 3: Follows template (now condensed)
- [ ] Phase 4: Follows template (now condensed)
- [ ] Phase 5: Follows template
- [ ] Phase 6: Follows template
- [ ] Phase 7: Follows template
- [ ] Phase 8: Follows template

**Time estimate**: 25 minutes

---

## IMPLEMENTATION CHECKLIST

Copy and paste into your task tracker:

```
TASK 1: Metadata Reconciliation (30 min)
  ☐ Open SKILL.md file
  ☐ Replace frontmatter lines 1-10
  ☐ Add all 6 scripts
  ☐ Add all 3 MCP tools with functions
  ☐ Add trigger keywords
  ☐ Add dependencies
  ☐ Verify changes

TASK 2: Add Data Flow Section (45 min)
  ☐ Insert new section after System Architecture
  ☐ Add phase-to-phase data flow diagram
  ☐ Add agent communication pattern section
  ☐ Add data format examples (3+ examples)
  ☐ Update Phase 1-8 with "Data Flow" section
  ☐ Verify links work

TASK 3: Refactor Long Prompts (2 hours)
  ☐ PART A: Phase 3 Refactor
    ☐ Find Phase 3 section (line ~207)
    ☐ Replace with condensed version (10-15 lines)
    ☐ Create phase3-pattern-retrieval-detailed.md
    ☐ Move all 110+ lines to detailed file
    ☐ Verify Phase 3 condensed but complete
  
  ☐ PART B: Phase 4 Refactor
    ☐ Find Phase 4 section (line ~260)
    ☐ Replace with condensed version (10-15 lines)
    ☐ Create phase4-safe-application-detailed.md
    ☐ Move all 90+ lines to detailed file
    ☐ Verify Phase 4 condensed but complete

TASK 4: Add Resource Verification & Schemas (1 hour)
  ☐ Create dogfooding-database-schema.md
  ☐ Add all 3 SQL CREATE TABLE statements
  ☐ Add initialization instructions
  ☐ Add MCP configuration section to SKILL.md
  ☐ Add pre-flight validation section
  ☐ Add health check commands
  ☐ Verify all paths correct

TASK 5: Standardize Instruction Format (30 min)
  ☐ Create instruction template section
  ☐ Define execution patterns (3+ patterns)
  ☐ Review all 8 phases
  ☐ Update to follow template
  ☐ Verify consistent voice (imperative)
  ☐ Verify all steps numbered
  ☐ Verify all criteria have checkmarks

FINAL VERIFICATION:
  ☐ Read entire SKILL.md - feels clear and complete?
  ☐ Frontmatter complete and accurate?
  ☐ All phases follow template?
  ☐ All scripts referenced?
  ☐ All MCP tools documented?
  ☐ All paths correct?
  ☐ Pre-flight checklist usable?
  ☐ Data flow clear?
  ☐ Long prompts refactored?

TOTAL TIME ESTIMATE: 4.5 hours
```

---

## VALIDATION AFTER FIXES

Once all P1 fixes are applied:

```bash
# 1. Run pre-flight check
C:\Users\17175\scripts\dogfood-continuous-improvement.bat --check-only

# 2. Execute dry-run
C:\Users\17175\scripts\dogfood-continuous-improvement.bat memory-mcp --dry-run

# 3. Verify documentation
  - Read SKILL.md front to back (should be clear)
  - Check all links work
  - Verify all paths valid

# 4. Manual test
  - Execute actual cycle on memory-mcp project
  - Document results
  - Compare with expected output
```

---

## ESTIMATED TIME BREAKDOWN

| Task | Est. Time | Priority |
|------|-----------|----------|
| 1. Metadata | 30 min | P1 |
| 2. Data Flow | 45 min | P1 |
| 3. Refactor Prompts | 2 hours | P1 |
| 4. Verification & Schemas | 1 hour | P1 |
| 5. Standardize Format | 30 min | P1 |
| **TOTAL** | **4.5 hours** | **Critical** |

---

## SUCCESS CRITERIA

After all P1 fixes, the skill should:

✅ Have complete and accurate frontmatter  
✅ Have clear data flow documentation  
✅ Have refactored prompts (readable, <15 lines each)  
✅ Have all resources documented and verified  
✅ Have consistent instruction format  
✅ Be ready for production use  
✅ Pass validation tests  

---

**Ready to start?** Begin with Task 1 (Metadata Reconciliation) - it's the quickest win and unblocks other work.

**Questions?** Reference the full audit document: `C:\Users\17175\docs\audit-phase3-continuous-improvement.md`
