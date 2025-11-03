# Example 1: Quality Detection Phase - Connascence Analysis

## Scenario Overview

**Context**: You have a Memory MCP server implementation (`memory-manager.js`) that's experiencing performance degradation and maintenance challenges. You suspect code quality issues but need systematic analysis.

**Goal**: Use Phase 1 (Quality Detection) to identify connascence violations and code quality issues automatically.

**Duration**: 30-60 seconds
**Agents**: `code-analyzer`, `reviewer`

---

## Step-by-Step Walkthrough

### Step 1: Trigger Quality Detection

**User Request**:
```
"Analyze code quality for memory-manager.js - check for connascence violations"
```

**Auto-Trigger Detection**:
- Keywords: "analyze code quality", "connascence"
- Skill invoked: `sop-dogfooding-quality-detection`

### Step 2: Agent Spawning (Automatic)

**Behind the Scenes**:
```javascript
// Skill spawns 2 agents concurrently via Task tool
[Single Message - Parallel Execution]:
  Task("Code Quality Analyzer",
    "Analyze memory-manager.js using Connascence MCP. Detect violations: God Objects, Parameter Bombs, Cyclomatic Complexity, Deep Nesting, Magic Literals.",
    "code-analyzer")

  Task("Quality Reviewer",
    "Review Connascence analysis results. Categorize violations by severity (CRITICAL/HIGH/MEDIUM). Prioritize fixes.",
    "reviewer")
```

### Step 3: Connascence Analysis Execution

**Code Analyzer Agent Actions**:
1. Reads `memory-manager.js` (982 lines)
2. Calls `mcp__connascence__analyze_file`
3. Receives violation report

**Sample Analysis Output**:
```json
{
  "file": "memory-manager.js",
  "violations": [
    {
      "type": "God Object",
      "severity": "CRITICAL",
      "line": 1,
      "details": "MemoryManager class has 34 methods (threshold: 15)",
      "methods": [
        "constructor", "initialize", "vectorStore", "storeMemory",
        "retrieveMemory", "searchSimilar", "updateMemory", "deleteMemory",
        "bulkStore", "bulkRetrieve", "clearExpired", "getStats",
        "exportMemories", "importMemories", "backup", "restore",
        "optimizeIndex", "rebuildIndex", "validateMemory", "sanitize",
        "encrypt", "decrypt", "compress", "decompress", "shard",
        "merge", "migrate", "healthCheck", "benchmark", "profile",
        "monitor", "alert", "log", "trace"
      ]
    },
    {
      "type": "Parameter Bomb (CoP)",
      "severity": "CRITICAL",
      "line": 156,
      "details": "storeMemory() has 12 parameters (NASA limit: 6)",
      "params": [
        "content", "tags", "metadata", "retention", "priority",
        "embedding", "timestamp", "source", "category", "project",
        "intent", "expiration"
      ]
    },
    {
      "type": "Cyclomatic Complexity",
      "severity": "HIGH",
      "line": 423,
      "details": "searchSimilar() complexity: 18 (threshold: 10)",
      "paths": 18
    },
    {
      "type": "Deep Nesting",
      "severity": "CRITICAL",
      "line": 589,
      "details": "handleQuery() has 9 nesting levels (NASA limit: 4)",
      "context": "if → try → if → for → if → switch → case → if → for"
    },
    {
      "type": "Long Function",
      "severity": "MEDIUM",
      "line": 234,
      "details": "vectorStore() has 127 lines (threshold: 50)",
      "lines": 127
    },
    {
      "type": "Magic Literal (CoM)",
      "severity": "MEDIUM",
      "line": 78,
      "details": "Hardcoded timeout: 30000",
      "value": "30000"
    },
    {
      "type": "Magic Literal (CoM)",
      "severity": "MEDIUM",
      "line": 145,
      "details": "Hardcoded port: 8080",
      "value": "8080"
    }
  ],
  "summary": {
    "total_violations": 7,
    "critical": 3,
    "high": 1,
    "medium": 3,
    "analysis_time": "0.042s"
  }
}
```

### Step 4: Quality Review & Prioritization

**Reviewer Agent Actions**:
1. Receives violation data from code-analyzer
2. Categorizes by impact and fix complexity
3. Creates prioritized action plan

**Review Output**:
```markdown
## Quality Review Report: memory-manager.js

### CRITICAL Issues (Fix Immediately)
1. **God Object** (Line 1)
   - Impact: Violates SRP, high coupling, maintenance nightmare
   - Fix Complexity: HIGH (requires class decomposition)
   - Recommendation: Extract 5-6 specialized classes

2. **Parameter Bomb** (Line 156)
   - Impact: NASA-critical violation, cognitive overload
   - Fix Complexity: MEDIUM (use options object)
   - Recommendation: Consolidate into MemoryOptions interface

3. **Deep Nesting** (Line 589)
   - Impact: NASA-critical violation, error-prone
   - Fix Complexity: MEDIUM (early returns, extract methods)
   - Recommendation: Reduce to ≤4 levels with guard clauses

### HIGH Priority
4. **Cyclomatic Complexity** (Line 423)
   - Fix: Extract search strategies into separate functions

### MEDIUM Priority
5. **Long Function** (Line 234) - Split into smaller functions
6-7. **Magic Literals** (Lines 78, 145) - Move to config constants

### Estimated Fix Time: 4-6 hours
```

### Step 5: Memory Storage (Automatic)

**What Gets Stored in Memory-MCP**:
```javascript
// Auto-tagged via memory-mcp-tagging-protocol.js
{
  content: "Connascence analysis for memory-manager.js detected 7 violations...",
  metadata: {
    // WHO
    agent: "code-analyzer",
    agent_category: "code-quality",
    agent_capabilities: ["connascence-detection", "code-analysis"],

    // WHEN
    timestamp: "2025-11-02T14:23:45.678Z",
    timestamp_unix: 1730559825,
    timestamp_readable: "November 2, 2025 2:23 PM",

    // PROJECT
    project: "memory-mcp-triple-system",
    file: "memory-manager.js",

    // WHY
    intent: "analysis",
    phase: "quality-detection",
    violations: ["God Object", "Parameter Bomb", "Deep Nesting", "Cyclomatic Complexity", "Long Function", "Magic Literals"],
    severity_breakdown: { critical: 3, high: 1, medium: 3 }
  },
  layer: "mid-term" // 7-day retention
}
```

---

## Code Example: Before Analysis

**memory-manager.js (God Object + Parameter Bomb)**:
```javascript
class MemoryManager {
  constructor(config) {
    this.db = null;
    this.vectorIndex = null;
    this.cache = new Map();
    // ... 30+ instance variables
  }

  // God Object: 34 methods in one class
  async initialize() { /* ... */ }
  async vectorStore() { /* 127 lines */ }
  async storeMemory() { /* ... */ }
  async retrieveMemory() { /* ... */ }
  async searchSimilar() { /* Complexity: 18 */ }
  async updateMemory() { /* ... */ }
  async deleteMemory() { /* ... */ }
  // ... 27 more methods

  // Parameter Bomb: 12 parameters (NASA limit: 6)
  async storeMemory(
    content,      // 1
    tags,         // 2
    metadata,     // 3
    retention,    // 4
    priority,     // 5
    embedding,    // 6
    timestamp,    // 7
    source,       // 8
    category,     // 9
    project,      // 10
    intent,       // 11
    expiration    // 12
  ) {
    // Deep Nesting: 9 levels
    if (content) {
      try {
        if (this.isValid(content)) {
          for (const tag of tags) {
            if (tag.type === 'critical') {
              switch (tag.action) {
                case 'store':
                  if (this.cache.has(tag.key)) {
                    for (const entry of this.cache.values()) {
                      if (entry.matches(tag)) {
                        // 9 levels deep!
                      }
                    }
                  }
                  break;
              }
            }
          }
        }
      } catch (err) { /* ... */ }
    }

    // Magic Literals
    const timeout = 30000; // No constant
    const port = 8080;     // Hardcoded
  }

  // Cyclomatic Complexity: 18 paths
  async searchSimilar(query, options) {
    if (!query) return null;
    if (options.mode === 'exact') { /* path 1 */ }
    else if (options.mode === 'fuzzy') { /* path 2 */ }
    else if (options.mode === 'semantic') {
      if (options.threshold > 0.8) { /* path 3 */ }
      else if (options.threshold > 0.6) { /* path 4 */ }
      else { /* path 5 */ }
    }
    // ... 13 more decision paths
  }
}
```

---

## Outcomes & Results

### Quantitative Metrics
- **Violations Detected**: 7 total (3 CRITICAL, 1 HIGH, 3 MEDIUM)
- **Analysis Time**: 0.042 seconds
- **File Size**: 982 lines
- **Methods Analyzed**: 34
- **NASA Violations**: 2 (Parameter Bomb, Deep Nesting)

### Qualitative Insights
- **God Object** identified as root cause of maintenance issues
- **Parameter Bomb** creating cognitive overload for developers
- **Deep Nesting** causing bugs in error handling logic
- All violations automatically stored in Memory-MCP with WHO/WHEN/PROJECT/WHY tags

### Memory-MCP Storage Benefits
- Future searches can retrieve this analysis instantly
- Pattern matching enables finding similar God Objects in other files
- Retention policy ensures 7-day availability for iterative fixes

---

## Tips & Best Practices

### ✅ DO:
1. **Run on modified files regularly** - Catch violations early
2. **Use as pre-commit hook** - Prevent violations from merging
3. **Store analysis results** - Build historical knowledge base
4. **Prioritize CRITICAL/HIGH** - Focus on NASA violations first
5. **Combine with Phase 2** - Search for proven fix patterns

### ❌ DON'T:
1. **Ignore NASA violations** - Parameter Bombs and Deep Nesting are safety-critical
2. **Fix everything at once** - Incremental refactoring reduces risk
3. **Skip memory storage** - Historical data enables pattern learning
4. **Analyze generated code** - Focus on human-written code first
5. **Dismiss Magic Literals** - They often indicate missing configuration

### Advanced Techniques

**1. Batch Analysis**:
```bash
# Analyze entire codebase
mcp__connascence__analyze_workspace --path ./src
```

**2. CI/CD Integration**:
```yaml
# .github/workflows/quality.yml
- name: Connascence Analysis
  run: |
    npx claude-flow skill sop-dogfooding-quality-detection
    if [ $CRITICAL_VIOLATIONS -gt 0 ]; then exit 1; fi
```

**3. Pre-Commit Hook**:
```bash
# hooks/pre-commit
#!/bin/bash
staged_files=$(git diff --cached --name-only --diff-filter=ACM | grep '.js$')
for file in $staged_files; do
  npx claude-flow skill sop-dogfooding-quality-detection --file "$file"
done
```

### Common Pitfalls

**Pitfall 1: False Positives**
- **Problem**: Not all 15+ method classes are God Objects
- **Solution**: Check cohesion - do methods share state and collaborate?

**Pitfall 2: Over-Optimization**
- **Problem**: Splitting every 51-line function
- **Solution**: Focus on functions with high complexity AND length

**Pitfall 3: Configuration Magic Literals**
- **Problem**: Moving hardcoded values to config without validation
- **Solution**: Use environment variables with schema validation

---

## Next Steps

After completing Phase 1 (Quality Detection):

1. **Review the violation report** - Understand impact and priority
2. **Proceed to Phase 2** (Pattern Retrieval) - Search Memory-MCP for proven fixes
3. **OR manually fix** - Use prioritization to guide refactoring
4. **Store fixes in Memory-MCP** - Enable future pattern matching

**Command to Trigger Phase 2**:
```
"Find fixes for God Object pattern in Memory-MCP"
```

---

## Summary

**Phase 1 Quality Detection** provides:
- ✅ Automated connascence violation detection (7 types)
- ✅ NASA compliance checking (Parameter Bombs, Deep Nesting)
- ✅ Severity-based prioritization (CRITICAL → MEDIUM)
- ✅ Automatic Memory-MCP storage with WHO/WHEN/PROJECT/WHY tags
- ✅ 30-60 second analysis time
- ✅ Foundation for Phase 2 pattern retrieval

**Performance**: 0.042s analysis time for 982-line file (23,000 lines/second)

**Integration**: Works seamlessly with git hooks, CI/CD pipelines, and manual workflows
