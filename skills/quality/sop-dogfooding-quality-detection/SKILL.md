---
name: sop-dogfooding-quality-detection
description: 3-part dogfooding workflow Phase 1 - Run Connascence analysis, store
  results in Memory-MCP with WHO/WHEN/PROJECT/WHY tagging. 30-60 seconds execution
  time.
agents: code-analyzer, reviewer
mcp_tools: connascence-analyzer, memory-mcp
scripts: dogfood-quality-check.bat, store-connascence-results.js
version: 1.0.0
category: quality
tags:
- quality
- testing
- validation
author: ruv
---

# SOP: Dogfooding Quality Detection

**Loop 1 of 3-Part System**: Code Quality Detection → Memory Storage

**Purpose**: Automatically detect code quality violations and store findings for cross-session learning

**Timeline**: 30-60 seconds

---

## System Architecture

```
[Code Generation]
    ↓
[Connascence Analysis] (7 violation types)
    ↓
[Memory-MCP Storage] (with WHO/WHEN/PROJECT/WHY tags)
    ↓
[Dashboard Update] (Grafana metrics)
```

---

## Phase 1: Pre-Analysis Health Check (5 sec)

**Agent**: `code-analyzer`

**Prompt**:
```javascript
await Task("Code Quality Checker", `
Check system health:
1. Verify Connascence Analyzer MCP is operational
   Command: cd C:\\Users\\17175\\Desktop\\connascence && python -m mcp.cli health-check
   Expected: {"status": "healthy"}

2. Verify Memory-MCP ChromaDB accessible
   Command: python -c "from src.indexing.vector_indexer import VectorIndexer; vi = VectorIndexer(); print('OK')"
   Expected: "OK"

3. Confirm target project exists
   Path: <project-directory>
   Check: Directory contains .py, .js, .ts files

Report status: READY or BLOCKED
`, "code-analyzer");
```

**MCP Tools Used**:
- `mcp__connascence-analyzer__health_check`

**Success Criteria**:
- Connascence returns "healthy"
- ChromaDB initialized
- Target has code files

---

## Phase 2: Run Connascence Analysis (15-30 sec)

**Agent**: `code-analyzer`

**Prompt**:
```javascript
await Task("Violation Detector", `
Run Connascence analysis on target project.

Target: <project-name>
Script: C:\\Users\\17175\\scripts\\dogfood-quality-check.bat <project-name>

Detection Types (7):
1. God Objects (threshold: >15 methods)
2. Parameter Bombs (NASA limit: >6 params)
3. Cyclomatic Complexity (threshold: >10)
4. Deep Nesting (NASA limit: >4 levels)
5. Long Functions (threshold: >50 lines)
6. Magic Literals (hardcoded values)
7. Duplicate Code Blocks

Output:
- JSON: C:\\Users\\17175\\metrics\\dogfooding\\<project>_<timestamp>.json
- Summary: C:\\Users\\17175\\metrics\\dogfooding\\summary_<timestamp>.txt

Count violations by type and severity (CRITICAL > HIGH > MEDIUM > LOW).
`, "code-analyzer");
```

**Script**: `C:\Users\17175\scripts\dogfood-quality-check.bat`

**MCP Tools Used**:
- `mcp__connascence-analyzer__analyze_workspace`

**Bash Commands**:
```bash
# Windows
C:\Users\17175\scripts\dogfood-quality-check.bat memory-mcp

# Runs internally:
cd C:\Users\17175\Desktop\connascence
python -m mcp.cli analyze-workspace C:\Users\17175\Desktop\memory-mcp-triple-system\src --output C:\Users\17175\metrics\dogfooding\memory-mcp_<timestamp>.json
```

**Output Example**:
```json
{
  "project": "memory-mcp-triple-system",
  "timestamp": "2025-11-02T12:00:00Z",
  "files_analyzed": 49,
  "violations": {
    "god_object": 2,
    "parameter_bomb": 3,
    "deep_nesting": 3,
    "cyclomatic_complexity": 3,
    "long_function": 19,
    "magic_literal": 15
  },
  "total_violations": 45,
  "critical_count": 8
}
```

**Success Criteria**:
- Analysis completes without errors
- JSON file generated
- Violations categorized by type

---

## Phase 3: Store in Memory-MCP (10-20 sec)

**Agent**: `code-analyzer`

**Prompt**:
```javascript
await Task("Memory Storage Agent", `
Store Connascence results in Memory-MCP with proper tagging.

Input JSON: C:\\Users\\17175\\metrics\\dogfooding\\<project>_<timestamp>.json

Script: node C:\\Users\\17175\\scripts\\store-connascence-results.js --project <name> --file <json-path>

Apply WHO/WHEN/PROJECT/WHY Protocol:

WHO:
- agent: "code-analyzer"
- agent_category: "code-quality"

WHEN:
- timestamp_iso: "<ISO-8601>"
- timestamp_unix: <unix-timestamp>
- timestamp_readable: "YYYY-MM-DD HH:MM:SS UTC"

PROJECT:
- project: "<project-name>"

WHY:
- intent: "code-quality-improvement"

ADDITIONAL:
- severity: "critical|high|medium|low"
- fix_category: "god-object|parameter-bomb|etc"
- violations_count: <number>
- platform: "windows|linux|mac"

Store via: VectorIndexer + EmbeddingPipeline
Collection: memory_chunks
Embedding: 384-dimensional (sentence-transformers)
`, "code-analyzer");
```

**Script**: `C:\Users\17175\scripts\store-connascence-results.js`

**MCP Tools Used**:
- `mcp__memory-mcp__memory_store` (via VectorIndexer)

**Node.js Script Logic**:
```javascript
// store-connascence-results.js
const { VectorIndexer } = require('../Desktop/memory-mcp-triple-system/src/indexing/vector_indexer');
const { EmbeddingPipeline } = require('../Desktop/memory-mcp-triple-system/src/indexing/embedding_pipeline');

// Load results JSON
const results = JSON.parse(fs.readFileSync(args.file));

// Generate text summary
const text = `
# Connascence Analysis: ${results.project}
Timestamp: ${results.timestamp}
Files Analyzed: ${results.files_analyzed}
Total Violations: ${results.total_violations}

CRITICAL (NASA Compliance):
- Parameter Bombs: ${results.violations.parameter_bomb}
- Deep Nesting: ${results.violations.deep_nesting}

Violations by Type:
${JSON.stringify(results.violations, null, 2)}
`;

// Create metadata
const metadata = {
  agent: "code-analyzer",
  agent_category: "code-quality",
  timestamp_iso: new Date().toISOString(),
  timestamp_unix: Math.floor(Date.now() / 1000),
  timestamp_readable: new Date().toUTCString(),
  project: results.project,
  intent: "code-quality-improvement",
  severity: results.critical_count > 0 ? "critical" : "medium",
  violations_count: results.total_violations
};

// Store in ChromaDB
const indexer = new VectorIndexer();
const embedder = new EmbeddingPipeline();
const embedding = embedder.encode_single(text);

indexer.collection.add({
  ids: [`dogfooding-${Date.now()}`],
  embeddings: [embedding.tolist()],
  documents: [text],
  metadatas: [metadata]
});
```

**Success Criteria**:
- Data stored in ChromaDB
- Metadata tags applied correctly
- Searchable via `vector_search`

---

## Phase 4: Generate Summary Report (5 sec)

**Agent**: `reviewer`

**Prompt**:
```javascript
await Task("Report Generator", `
Generate human-readable summary from analysis results.

Input: C:\\Users\\17175\\metrics\\dogfooding\\<project>_<timestamp>.json

Output Format:
============================================================
CONNASCENCE ANALYSIS SUMMARY
============================================================
Project: <name>
Timestamp: <ISO>
Files Analyzed: <count>
Total Violations: <count>

CRITICAL - NASA Compliance Violations (Fix Immediately):
- Parameter Bombs: <count> files (>6 params, NASA limit exceeded)
- Deep Nesting: <count> files (>4 levels, NASA limit exceeded)

HIGH - Code Quality Issues (Refactor Soon):
- God Objects: <count> files (>15 methods)
- Cyclomatic Complexity: <count> files (>10)

MEDIUM - Maintenance (Refactor When Possible):
- Long Functions: <count> files (>50 lines)
- Magic Literals: <count> files (hardcoded values)

RECOMMENDATIONS:
1. Address NASA violations first (Parameter Bombs, Deep Nesting)
2. Refactor God Objects using Delegation Pattern
3. Extract Magic Literals to named constants
4. Run connascence-dogfooding-pattern-retrieval to find similar fixes

Next Action: Query Memory-MCP for fix patterns
Command: C:\\Users\\17175\\scripts\\dogfood-memory-retrieval.bat "fix parameter bomb"
============================================================

Save to: C:\\Users\\17175\\metrics\\dogfooding\\summary_<timestamp>.txt
`, "reviewer");
```

**Success Criteria**:
- Summary file created
- Violations prioritized by severity
- Actionable recommendations included

---

## Phase 5: Dashboard Update & Coordination (5 sec)

**Agent**: `code-analyzer`

**Prompt**:
```javascript
await Task("Metrics Coordinator", `
Update Grafana dashboard and notify completion.

Actions:
1. Insert analysis record into SQLite DB
   DB: C:\\Users\\17175\\metrics\\dogfooding\\dogfooding.db
   Table: violations
   Columns: project, timestamp, file_count, total_violations, critical_count, high_count, medium_count

2. Trigger dashboard refresh
   Endpoint: http://localhost:3000/api/datasources/proxy/1/refresh

3. Send MCP coordination hook
   Command: npx claude-flow@alpha hooks post-task --task-id "quality-detection-<timestamp>" --status "complete" --violations "<count>"

4. Prepare next phase trigger
   Set flag: READY_FOR_PATTERN_RETRIEVAL=true

Store completion status in memory: dogfooding/quality-detection/status
`, "code-analyzer");
```

**MCP Tools Used**:
- Claude Flow hooks system

**Bash Commands**:
```bash
# Update SQLite DB
sqlite3 C:\Users\17175\metrics\dogfooding\dogfooding.db \
  "INSERT INTO violations (project, timestamp, total_violations) VALUES ('<project>', '<timestamp>', <count>)"

# Trigger hook
npx claude-flow@alpha hooks post-task --task-id "qd-<timestamp>" --status "complete"
```

**Success Criteria**:
- Database updated
- Dashboard refreshed
- Hook notification sent
- Ready for Phase 2 (pattern retrieval)

---

## Error Handling

### If Connascence Analysis Fails:

**Agent**: `code-analyzer`

```javascript
await Task("Error Handler", `
Diagnosis:
1. Check Connascence Analyzer health
   Command: cd C:\\Users\\17175\\Desktop\\connascence && python -m mcp.cli health-check

2. Check Python environment
   Command: python --version (expected: 3.12+)

3. Check virtual environment activated
   Path: C:\\Users\\17175\\Desktop\\connascence\\venv-connascence

4. Re-run with verbose logging
   Command: python -m mcp.cli analyze-workspace <project> --verbose

If still failing:
- Store error in Memory-MCP with intent: "error-diagnosis"
- Roll back to last known good state
- Alert user with error details
`, "code-analyzer");
```

### If Memory-MCP Storage Fails:

**Agent**: `code-analyzer`

```javascript
await Task("Storage Error Handler", `
Diagnosis:
1. Verify ChromaDB accessible
   Command: python -c "from src.indexing.vector_indexer import VectorIndexer; vi = VectorIndexer(); print(vi.collection.count())"

2. Check disk space
   Command: df -h (Linux/Mac) or wmic logicaldisk get size,freespace (Windows)

3. Verify VectorIndexer.collection initialized
   Fix: Ensure VectorIndexer.__init__ calls create_collection()

4. Retry storage with exponential backoff
   Attempts: 3
   Delay: 1s, 2s, 4s

If persistent failure:
- Store analysis locally: C:\\Users\\17175\\metrics\\dogfooding\\pending\\<timestamp>.json
- Queue for retry when Memory-MCP recovers
`, "code-analyzer");
```

---

## Metrics Tracked

1. **Analysis Duration**: Seconds (target: <30s)
2. **Violations Detected**: Count by type
3. **Files Analyzed**: Total count
4. **Storage Success Rate**: Percentage (target: 100%)
5. **Dashboard Update Status**: Success/Fail

---

## Integration with 3-Part System

**Current Phase**: Phase 1 (Quality Detection)

**Triggers Next Phase**:
- `sop-dogfooding-pattern-retrieval` - Query Memory-MCP for similar fixes

**Triggered By**:
- `functionality-audit` - After code generation
- `code-review-assistant` - During PR review
- `production-readiness` - Before deployment
- Manual trigger: `dogfood-quality-check.bat <project>`

**Works With**:
- `connascence-analyzer` MCP - Analysis engine
- `memory-mcp` - Storage system
- `code-analyzer` agent - Primary executor
- `reviewer` agent - Report generation

---

## Quick Reference

```bash
# Single project
C:\Users\17175\scripts\dogfood-quality-check.bat memory-mcp

# All projects
C:\Users\17175\scripts\dogfood-quality-check.bat all

# Expected outputs:
# 1. JSON: metrics/dogfooding/<project>_<timestamp>.json
# 2. Summary: metrics/dogfooding/summary_<timestamp>.txt
# 3. Memory-MCP storage confirmation
# 4. Dashboard update
```

**Total Duration**: 30-60 seconds
**Agents**: `code-analyzer` (primary), `reviewer` (secondary)
**Tools**: Connascence Analyzer MCP, Memory-MCP, Bash, SQLite
**Output**: Violations detected + stored with metadata + dashboard updated

---

## Safety Rules (CRITICAL)

**From**: `C:\Users\17175\docs\DOGFOODING-SAFETY-RULES.md`

1. **Sandbox Testing REQUIRED** before applying any fixes
2. **Automated Rollback** via git stash
3. **Progressive Application** (one fix at a time)
4. **Test Coverage ≥70%** required
5. **CI/CD Gate** must pass before merge

**See**: Full safety rules documentation for details