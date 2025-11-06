---
name: sop-dogfooding-pattern-retrieval
description: 3-part dogfooding workflow Phase 2 - Query Memory-MCP for similar past fixes using vector search, rank patterns, optionally apply transformations. 10-30 seconds execution time.
agents: code-analyzer, coder, reviewer
mcp_tools: memory-mcp
scripts: dogfood-memory-retrieval.bat, query-memory-mcp.js, apply-fix-pattern.js
---

# SOP: Dogfooding Pattern Retrieval

**Loop 2 of 3-Part System**: Pattern Retrieval → Application

**Purpose**: Query Memory-MCP for similar successful fixes and apply best practices

**Timeline**: 10-30 seconds

---

## System Architecture

```
[Violation Detected] (from Phase 1)
    ↓
[Vector Search Memory-MCP] (semantic similarity)
    ↓
[Rank & Select Best Pattern] (similarity + success rate)
    ↓
[Apply Pattern (Optional)] (AST transformation)
    ↓
[Store Application Result] (success/failure metrics)
```

---

## Phase 1: Identify Fix Context (5 sec)

**Agent**: `code-analyzer`

**Prompt**:
```javascript
await Task("Context Analyzer", `
Extract violation context for semantic search.

Input from Phase 1 (Quality Detection):
- Violation type: <type> (e.g., "God Object", "Parameter Bomb")
- File path: <path>
- Violation details: <details> (e.g., "26 methods", "10 parameters")
- Severity: <critical|high|medium|low>

Formulate semantic search query:

Examples:
- "How to fix God Object with 26 methods"
- "Refactor Parameter Bomb with 10 parameters to meet NASA limit"
- "Reduce Deep Nesting from 6 levels to 4"
- "Extract Magic Literals to named constants"
- "Break long function into smaller methods"

Query Construction Rules:
1. Include violation type
2. Include quantitative metric (e.g., "26 methods")
3. Include desired outcome (e.g., "to 4 levels")
4. Use natural language (for better embedding similarity)

Store query: dogfooding/pattern-retrieval/query-<timestamp>
`, "code-analyzer");
```

**Success Criteria**:
- Query formulated with context
- Violation type + metrics included
- Natural language phrasing

---

## Phase 2: Vector Search Memory-MCP (5-10 sec)

**Agent**: `code-analyzer`

**Prompt**:
```javascript
await Task("Pattern Searcher", `
Execute vector search on Memory-MCP to find similar past fixes.

Query: "<formulated-query>"
Script: node C:\\Users\\17175\\scripts\\query-memory-mcp.js --query "<query>" --limit 5 --output query-<timestamp>.json

Vector Search Process:
1. Generate query embedding (sentence-transformers, 384-dim)
2. Search ChromaDB collection 'memory_chunks'
3. Use HNSW index with cosine similarity
4. Rank by distance (closer = more similar)
5. Filter by metadata:
   - intent: "bugfix" OR "code-quality-improvement"
   - project: <any>
   - agent_category: "code-quality"
6. Return top 5 matches with similarity scores

Similarity Scoring:
- score = 1 - distance
- Excellent: score ≥ 0.85
- Good: score ≥ 0.70
- Fair: score ≥ 0.50
- Poor: score < 0.50

Output: C:\\Users\\17175\\metrics\\dogfooding\\retrievals\\query-<timestamp>.json
`, "code-analyzer");
```

**Script**: `C:\Users\17175\scripts\query-memory-mcp.js`

**Node.js Script Logic**:
```javascript
// query-memory-mcp.js
const { VectorIndexer } = require('../Desktop/memory-mcp-triple-system/src/indexing/vector_indexer');
const { EmbeddingPipeline } = require('../Desktop/memory-mcp-triple-system/src/indexing/embedding_pipeline');

// Initialize
const indexer = new VectorIndexer();
const embedder = new EmbeddingPipeline();

// Generate query embedding
const queryEmbedding = embedder.encode_single(args.query);

// Search ChromaDB
const results = indexer.collection.query({
  query_embeddings: [queryEmbedding.tolist()],
  n_results: args.limit,
  where: {
    "$or": [
      {"intent": "bugfix"},
      {"intent": "code-quality-improvement"}
    ]
  }
});

// Format results
const formatted = results.ids[0].map((id, i) => ({
  id,
  text: results.documents[0][i],
  metadata: results.metadatas[0][i],
  distance: results.distances[0][i],
  similarity: 1 - results.distances[0][i]
}));

// Save to file
fs.writeFileSync(args.output, JSON.stringify(formatted, null, 2));
```

**MCP Tools Used**:
- `mcp__memory-mcp__vector_search` (via VectorIndexer)

**Output Example**:
```json
[
  {
    "id": "dogfooding-abc123",
    "text": "# Fixed God Object with 26 methods\nApplied Delegation Pattern...",
    "metadata": {
      "agent": "bugfix-agent",
      "project": "connascence-analyzer",
      "intent": "bugfix",
      "fix_category": "god-object",
      "before": "26 methods",
      "after": "12 methods",
      "improvement": "54% reduction"
    },
    "distance": 0.13,
    "similarity": 0.87
  },
  {...},
  {...}
]
```

**Success Criteria**:
- ≥3 relevant results found
- Top result similarity ≥0.70
- Metadata includes transformation details

---

## Phase 3: Analyze Retrieved Patterns (5-10 sec)

**Agent**: `code-analyzer` + `reviewer`

**Prompt**:
```javascript
await Task("Pattern Analyzer", `
Analyze top 3 retrieved patterns for applicability.

Input: C:\\Users\\17175\\metrics\\dogfooding\\retrievals\\query-<timestamp>.json

For each pattern:
1. Extract transformation type:
   - Delegation Pattern (God Object → separate classes)
   - Config Object Pattern (Parameter Bomb → object param)
   - Early Return Pattern (Deep Nesting → guard clauses)
   - Extract Method Pattern (Long Function → smaller functions)
   - Named Constant Pattern (Magic Literal → const)
   - Extract Function Pattern (Duplicate Code → DRY)

2. Assess context similarity:
   - Same violation type? (YES/NO)
   - Similar complexity? (e.g., both 20+ methods)
   - Same language? (Python, TypeScript, etc.)
   - Similar domain? (web server, data processing, etc.)

3. Extract improvement metrics:
   - Before state (e.g., "26 methods")
   - After state (e.g., "12 methods")
   - Improvement percentage (e.g., "54% reduction")
   - Side effects (e.g., "None", "Required dependency injection")

4. Check success indicators:
   - Tests still passing after fix?
   - Code quality score improved?
   - No new violations introduced?

Output analysis report for top 3 patterns.
`, "code-analyzer");
```

**Pattern Type Examples**:

**1. Delegation Pattern** (God Object fix):
```javascript
// Before: 26 methods
class Processor {
  process() {...}
  validate() {...}
  transform() {...}
  save() {...}
  // ... 22 more methods
}

// After: 12 methods (54% reduction)
class Processor {
  constructor() {
    this.validator = new Validator(); // 5 methods
    this.transformer = new Transformer(); // 6 methods
    this.storage = new Storage(); // 3 methods
  }
  process() {
    this.validator.validate(...);
    this.transformer.transform(...);
    this.storage.save(...);
  }
}
```

**2. Config Object Pattern** (Parameter Bomb fix):
```javascript
// Before: 10 parameters (violates NASA limit of 6)
function request(url, method, headers, body, timeout, retries, cache, auth, proxy, debug) {...}

// After: 1 parameter
function request(config) {
  const { url, method, headers, body, timeout, retries, cache, auth, proxy, debug } = config;
  ...
}
```

**Success Criteria**:
- Transformation type identified
- Context similarity assessed
- Improvement metrics extracted
- Applicability determined

---

## Phase 4: Rank & Select Best Pattern (5 sec)

**Agent**: `reviewer`

**Prompt**:
```javascript
await Task("Pattern Ranker", `
Rank retrieved patterns and select best match.

Ranking Algorithm (weighted scoring):

rank_score = (
  similarity * 0.40 +                    // Vector similarity
  success_rate * 0.30 +                  // Historical success
  context_match * 0.20 +                 // Same violation type
  recency_bonus * 0.10                   // Recent fixes preferred
)

Components:
1. Similarity: From vector search (0-1)
2. Success Rate: From metadata.success_rate or default 0.75
3. Context Match: 1.0 if same violation type, 0.5 if related, 0 if different
4. Recency Bonus: 1.0 if <7 days, 0.75 if <30 days, 0.5 if <90 days, 0.25 if older

Apply ranking to all retrieved patterns.

Select top-ranked pattern.

Output:
{
  "selected_pattern": {
    "id": "dogfooding-abc123",
    "rank_score": 0.85,
    "similarity": 0.87,
    "transformation": "delegation",
    "fix_category": "god-object",
    "confidence": "high",
    "recommendation": "Apply Delegation Pattern - extract 14 methods to 2 new classes"
  }
}

Save to: C:\\Users\\17175\\metrics\\dogfooding\\retrievals\\best-pattern-<timestamp>.json
`, "reviewer");
```

**Ranking Example**:
```javascript
Pattern A:
  similarity: 0.87
  success_rate: 0.90
  context_match: 1.0 (exact match)
  recency_bonus: 1.0 (3 days old)
  rank_score = 0.87*0.4 + 0.90*0.3 + 1.0*0.2 + 1.0*0.1 = 0.918 ✓ BEST

Pattern B:
  similarity: 0.92
  success_rate: 0.60
  context_match: 0.5 (related)
  recency_bonus: 0.25 (120 days old)
  rank_score = 0.92*0.4 + 0.60*0.3 + 0.5*0.2 + 0.25*0.1 = 0.733
```

**Success Criteria**:
- All patterns ranked
- Top pattern selected
- Confidence level assigned
- Recommendation generated

---

## Phase 5: Apply Pattern (OPTIONAL, 10-30 sec)

**Agent**: `coder`

**Prompt**:
```javascript
await Task("Pattern Applicator", `
Apply selected pattern transformation to target code.

⚠️ SAFETY RULES CRITICAL ⚠️
See: C:\\Users\\17175\\docs\\DOGFOODING-SAFETY-RULES.md

MANDATORY STEPS:

1. Backup original file
   Command: cp <file> <file>.backup-<timestamp>

2. Parse target code (AST)
   Tool: @babel/parser or python ast module

3. Apply transformation
   Pattern: <selected-transformation>
   Target: <file-path>

4. Write transformed code

5. Run tests (CRITICAL)
   Command: npm test (or pytest)

6. If tests FAIL:
   ROLLBACK: mv <file>.backup-<timestamp> <file>
   ABORT: Do NOT commit changes

7. If tests PASS:
   Proceed to verification

Script: node C:\\Users\\17175\\scripts\\apply-fix-pattern.js --input best-pattern-<timestamp>.json --file <target> --rank 1

NEVER skip testing! ALWAYS rollback on failure!
`, "coder");
```

**Script**: `C:\Users\17175\scripts\apply-fix-pattern.js`

**Transformation Strategies**:

**Strategy 1: Delegation** (God Object):
```javascript
// AST transformation
function applyDelegation(ast, methodsToExtract) {
  // 1. Identify methods to extract
  // 2. Create new class definitions
  // 3. Move methods to new classes
  // 4. Add dependency injection in constructor
  // 5. Update method calls to use delegates
}
```

**Strategy 2: Config Object** (Parameter Bomb):
```javascript
// AST transformation
function applyConfigObject(ast, functionNode) {
  // 1. Find function with >6 params
  // 2. Create config object type/interface
  // 3. Replace params with single config param
  // 4. Add destructuring in function body
  // 5. Update all call sites
}
```

**Safety Checks**:
```bash
# Before transformation
git stash push -u -m "pre-pattern-application-<timestamp>"

# Apply transformation
node apply-fix-pattern.js --file <target> --pattern <id>

# Test
npm test

# If tests fail
if [ $? -ne 0 ]; then
  echo "ROLLBACK: Tests failed"
  git stash pop
  exit 1
fi

# If tests pass
git stash drop
git add <file>
git commit -m "Applied <pattern> - <improvement>"
```

**Success Criteria**:
- Transformation applied successfully
- Tests pass (100% required)
- No new violations introduced
- Code quality improved

---

## Phase 6: Store Application Result (5 sec)

**Agent**: `code-analyzer`

**Prompt**:
```javascript
await Task("Result Recorder", `
Record pattern application result in Memory-MCP.

Metadata:
{
  agent: "coder",
  agent_category: "code-quality",
  project: "<project-name>",
  intent: "pattern-application",

  // Pattern details
  pattern_id: "<id>",
  transformation: "<type>",
  applied_to: "<file-path>",

  // Result
  result: "success" | "failure",
  tests_passed: true | false,

  // Metrics
  before_violations: <count>,
  after_violations: <count>,
  improvement_percentage: <percent>,
  execution_time_ms: <ms>
}

Text Summary:
"Applied <transformation> to <file>:
- Before: <before-state>
- After: <after-state>
- Improvement: <percentage>%
- Tests: PASSED
- Violations Reduced: <count>"

Store via VectorIndexer for future retrievals.

Update success_rate for this pattern:
  new_success_rate = (old_success_rate * old_count + new_result) / (old_count + 1)
`, "code-analyzer");
```

**MCP Tools Used**:
- `mcp__memory-mcp__memory_store` (via VectorIndexer)

**Success Criteria**:
- Application result stored
- Success rate updated
- Metrics recorded
- Searchable for future use

---

## Error Handling

### If No Patterns Found:

**Agent**: `code-analyzer`

```javascript
await Task("Fallback Search", `
No patterns found. Broaden search.

Strategies:
1. Remove quantitative metrics from query
   Before: "God Object with 26 methods"
   After: "God Object refactoring"

2. Search by category only
   Query: "code quality improvement"

3. Increase result limit
   From: --limit 5
   To: --limit 10

4. Check total vector count in Memory-MCP
   Command: python -c "from src.indexing.vector_indexer import VectorIndexer; vi = VectorIndexer(); print(f'Total vectors: {vi.collection.count()}')"

5. If count is 0:
   ERROR: Memory-MCP has no stored patterns
   Action: Run Phase 1 (Quality Detection) first to populate data

Store "no patterns found" event for analysis.
`, "code-analyzer");
```

### If Pattern Application Fails:

**Agent**: `coder`

```javascript
await Task("Failure Handler", `
Pattern application failed. Execute rollback and analysis.

Immediate Actions:
1. Rollback changes
   Command: git checkout <file>

2. Verify rollback
   Command: git diff <file> (should be empty)

3. Re-run tests
   Command: npm test (should pass)

Root Cause Analysis:
1. Why did transformation fail?
   - Syntax error in generated code?
   - Tests failed due to logic change?
   - New violations introduced?

2. Record failure in Memory-MCP
   Metadata:
   {
     intent: "pattern-application-failure",
     pattern_id: "<id>",
     failure_reason: "<reason>",
     transformation: "<type>",
     DO_NOT_APPLY: true
   }

3. Update pattern success_rate
   Decrement on failure

4. Alert user
   Message: "Pattern application failed: <reason>. Changes rolled back. Tests passing."

Never leave codebase in broken state!
`, "coder");
```

---

## Metrics Tracked

1. **Query Time**: Milliseconds for vector search (target: <1000ms)
2. **Results Found**: Count (target: ≥3)
3. **Top Similarity Score**: 0-1 (target: ≥0.70)
4. **Application Success Rate**: Percentage (target: ≥95%)
5. **Improvement Magnitude**: Before/after comparison

---

## Integration with 3-Part System

**Current Phase**: Phase 2 (Pattern Retrieval)

**Triggered By**:
- `sop-dogfooding-quality-detection` - After violations detected

**Triggers Next**:
- `sop-dogfooding-continuous-improvement` - Full cycle execution

**Works With**:
- `memory-mcp` - Vector search
- `code-analyzer` agent - Pattern analysis
- `coder` agent - Pattern application
- `reviewer` agent - Ranking and selection

---

## Quick Reference

```bash
# Single query
C:\Users\17175\scripts\dogfood-memory-retrieval.bat "God Object with 26 methods"

# With optional application
C:\Users\17175\scripts\dogfood-memory-retrieval.bat "Parameter Bomb 10 params" --apply

# Expected outputs:
# 1. JSON: retrievals/query-<timestamp>.json
# 2. Best pattern: retrievals/best-pattern-<timestamp>.json
# 3. (Optional) Applied transformation
```

**Total Duration**: 10-30 seconds
**Agents**: `code-analyzer`, `coder`, `reviewer`
**Tools**: Memory-MCP, Bash, AST parser
**Output**: Best fix pattern + optional application + result storage
