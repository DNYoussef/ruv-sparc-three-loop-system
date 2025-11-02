# Root Cause Analysis Techniques

Systematic approaches for identifying true root causes in CI/CD failures, preventing symptom-focused fixes.

## 5-Whys Methodology

### Overview
Iteratively ask "Why?" to drill down from symptoms to root causes. Originated from Toyota Production System.

### Process
```
Problem: Test failed with "TypeError: Cannot read property 'verify' of undefined"

Why 1: Why did the test fail?
→ jwt.verify is undefined

Why 2: Why is jwt.verify undefined?
→ Import changed in jsonwebtoken package

Why 3: Why did the import change?
→ Package upgraded from v8 to v9 (breaking change)

Why 4: Why wasn't the breaking change handled?
→ No await on async verify() call

Why 5: Why was there no await?
→ Changelog not reviewed during upgrade

TRUE ROOT CAUSE: Dependency upgraded without reviewing breaking changes
```

### Best Practices
1. **Stop at True Root**: Don't stop at symptoms (e.g., "jwt.verify is undefined")
2. **Ask 5 Times Minimum**: Usually takes 4-6 "Why?" to reach true root
3. **Validate Each Answer**: Ensure each "Why?" answer is factually correct
4. **Avoid Blame**: Focus on systemic issues, not individual mistakes
5. **Document Reasoning**: Capture full 5-Whys chain for Loop 1 feedback

### Common Pitfalls
- **Stopping Too Early**: "Why did test fail? → Code is broken" (symptom, not root)
- **Multiple Root Causes**: If 5-Whys branches, you may have multiple roots
- **Speculation**: Base answers on evidence, not assumptions

## Failure Dependency Graph Analysis

### Overview
Model failures as directed graphs to identify root causes (0 incoming edges) and cascaded failures.

### Graph Construction

#### Nodes
```json
{
  "id": "jwt-verify-error",
  "type": "root" | "cascade",
  "file": "src/auth/jwt.js",
  "line": 42,
  "error": "TypeError: Cannot read property 'verify' of undefined",
  "in_degree": 0
}
```

#### Edges
```json
{
  "from": "jwt-verify-error",
  "to": "token-refresh-error",
  "relationship": "causes",
  "evidence": "token-refresh calls jwt.verify which is broken"
}
```

### Graph Algorithms

#### Topological Sort
Identifies root causes (nodes with in_degree = 0).

```javascript
function findRootCauses(graph) {
  return graph.nodes.filter(node => node.in_degree === 0);
}
```

#### Cascade Depth Calculation
Determines how many levels of failures cascade from each root.

```javascript
function calculateCascadeDepth(graph, rootNode) {
  const visited = new Set();
  let maxDepth = 0;

  function dfs(node, depth) {
    if (visited.has(node.id)) return;
    visited.add(node.id);
    maxDepth = Math.max(maxDepth, depth);

    for (const edge of graph.edges.filter(e => e.from === node.id)) {
      const childNode = graph.nodes.find(n => n.id === edge.to);
      dfs(childNode, depth + 1);
    }
  }

  dfs(rootNode, 0);
  return maxDepth;
}
```

#### Strongly Connected Components
Identifies circular dependencies (A depends on B, B depends on A).

```javascript
function findCircularDependencies(graph) {
  // Tarjan's algorithm or Kosaraju's algorithm
  // Returns sets of nodes that form cycles
}
```

### Example Analysis

**15 Test Failures → 2 Root Causes**

```
Graph:
  jwt-verify (root, in_degree=0) → 8 cascaded failures
  db-connect (root, in_degree=0) → 5 cascaded failures
  integration-flow (cascade, in_degree=2) ← depends on both roots

Insight:
  - Fix 2 root causes → Resolve 15 total failures
  - Cascade ratio: 13/15 = 86.7% (high leverage)
```

## Connascence Analysis

### Overview
Connascence measures coupling strength between code components. High connascence indicates fixes must be bundled atomically.

### Connascence Types

#### 1. Connascence of Name (CoN)
**Definition**: Multiple components must agree on the name of something.

**Example**:
```typescript
// File A: Define function
function verifyToken(token: string) { ... }

// Files B, C, D: Call function
const result = verifyToken(token); // All 3 files depend on name "verifyToken"
```

**Impact on Fixes**: Renaming `verifyToken` requires updating A, B, C, D **atomically**.

#### 2. Connascence of Type (CoT)
**Definition**: Multiple components must agree on the type of something.

**Example**:
```typescript
// File A: Change parameter type
function findById(id: number) { ... } // Was: id: string

// Files B, C, D: Call with wrong type
const user = findById(req.params.id); // ERROR: string not assignable to number
```

**Impact on Fixes**: Type changes require updating all callers **atomically**.

#### 3. Connascence of Algorithm (CoA)
**Definition**: Multiple components must agree on a particular algorithm.

**Example**:
```typescript
// Files A, B, C: All use same password hashing algorithm
const hash = bcrypt.hashSync(password, 10); // saltRounds=10

// If we change algorithm, must update A, B, C atomically
```

**Impact on Fixes**: Algorithm changes require updating all implementations **atomically**.

### Connascence-Aware Fix Strategies

#### Isolated Fix (Low Connascence)
```json
{
  "fixStrategy": "isolated",
  "reason": "Change affects single file only",
  "files": ["src/config/redis.ts"],
  "risk": "low"
}
```

#### Bundled Fix (High Connascence)
```json
{
  "fixStrategy": "bundled",
  "reason": "Type change affects 15 callers (Connascence of Type)",
  "files": [
    "src/services/user.service.ts",
    "src/controllers/user.controller.ts",
    "... 13 more files"
  ],
  "risk": "high - Must apply atomically or cascade failures"
}
```

#### Architectural Fix (Very High Connascence)
```json
{
  "fixStrategy": "architectural",
  "reason": "Breaking circular dependency requires refactoring",
  "files": [
    "Refactor: Extract shared logic to new module",
    "Update: All dependent modules"
  ],
  "risk": "very high - Requires design change"
}
```

## Error Propagation Tracing

### Overview
Trace error propagation from point of failure to observable symptoms using stack traces and call graphs.

### Propagation Patterns

#### 1. Direct Propagation
```
Error occurs → Immediately observable

Example:
  jwt.verify() throws → Test fails immediately
```

#### 2. Indirect Propagation
```
Error occurs → Sets bad state → Later code fails

Example:
  db.connect() fails silently → Routes loaded → API calls fail (not db.connect)
```

#### 3. Cascade Propagation
```
Error A → Causes Error B → Causes Error C

Example:
  Auth fails → Middleware fails → API fails → Integration fails
```

### Stack Trace Analysis

#### Identify Direct Cause
```
Error: Cannot read property 'verify' of undefined
  at verifyToken (src/auth/jwt.js:42:23)     ← Direct cause
  at authenticate (src/middleware/auth.js:15)
  at router (src/api/routes.js:8)
```

**Direct Cause**: `src/auth/jwt.js:42` - `jwt.verify` is undefined

#### Find Indirect Cause
```
Why is jwt.verify undefined?
  → Import changed: const jwt = require('jsonwebtoken')
  → jsonwebtoken v9 changed exports: verify is now async
  → No await used on async function
```

**Indirect Cause**: Package upgrade + missing await

### State-Based Error Tracing

#### Pattern
```
Bad State Set → ... → Bad State Used → Error

Example:
1. server.js: db.connect() (no await) → Connection not ready
2. routes.js: Load routes (depends on db) → Routes registered
3. API call: Try to query DB → ERROR: Connection not established
```

**Root Cause**: Step 1 (async not awaited)
**Observable Symptom**: Step 3 (connection error)

## Byzantine Consensus for Root Cause Validation

### Overview
Use multiple independent agents to validate root cause claims. Requires 5/7 agreement (Byzantine fault tolerance).

### Consensus Process

#### Step 1: Independent Analysis
7 agents analyze failures independently:
- 2 Researchers (external patterns)
- 1 Error Analyzer (stack traces)
- 1 Code Context Investigator (code analysis)
- 2 Test Auditors (test validity)
- 1 Dependency Detector (package analysis)

#### Step 2: Consensus Voting
For each root cause claim:
```json
{
  "claim": "JWT verification broken due to async breaking change",
  "votes": {
    "researcher-1": "agree",
    "researcher-2": "agree",
    "error-analyzer": "agree",
    "code-context": "agree",
    "test-auditor-1": "agree",
    "test-auditor-2": "agree",
    "dependency-detector": "agree"
  },
  "consensus": "7/7",
  "verdict": "CONFIRMED"
}
```

#### Step 3: Conflict Resolution
If < 5/7 agreement:
```json
{
  "claim": "Database migration failed due to state mismatch",
  "votes": {
    "agent-1": "agree",
    "agent-2": "agree",
    "agent-3": "disagree (thinks it's a permissions issue)",
    "agent-4": "agree",
    "agent-5": "disagree",
    "agent-6": "agree",
    "agent-7": "agree"
  },
  "consensus": "5/7",
  "verdict": "DISPUTED - Spawn tiebreaker agent for manual review"
}
```

### Confidence Scoring

#### High Confidence (7/7 or 6/7)
```json
{
  "consensus": "7/7",
  "confidence": "high",
  "action": "Proceed with fix based on this root cause"
}
```

#### Medium Confidence (5/7)
```json
{
  "consensus": "5/7",
  "confidence": "medium",
  "action": "Proceed with caution, add extra validation"
}
```

#### Low Confidence (< 5/7)
```json
{
  "consensus": "4/7",
  "confidence": "low",
  "action": "BLOCK - Manual review required, root cause unclear"
}
```

## Raft Consensus for Final Validation

### Overview
Leader-based consensus for final root cause list. Graph Analyst 2 (most validated data) is leader.

### Consensus Process

#### Step 1: Leader Election
```
Graph Analyst 2 → Leader (most comprehensive validation)
Graph Analyst 1 → Follower
Root Cause Validator → Follower
```

#### Step 2: Log Replication
Leader proposes root cause list → Followers validate against their data.

```json
{
  "leader_proposal": [
    "Root cause 1: JWT async breaking change",
    "Root cause 2: Database initialization not awaited"
  ],
  "follower_validations": {
    "graph-analyst-1": "AGREE (graph analysis confirms)",
    "root-validator": "AGREE (5-Whys confirms)"
  },
  "majority": "3/3",
  "verdict": "APPROVED"
}
```

#### Step 3: Conflict Resolution
If followers disagree:
```json
{
  "leader_proposal": "Root cause: Type mismatch",
  "follower_validations": {
    "graph-analyst-1": "AGREE",
    "root-validator": "DISAGREE - 5-Whys reveals deeper cause (incomplete migration)"
  },
  "resolution": "Override leader proposal with 5-Whys deeper cause"
}
```

## Integration with Loop 3

### Pre-Analysis
```bash
# Step 1: GitHub Hook Integration
# Download failure logs, parse structured data

# Step 2: AI-Powered Analysis
# Gemini large-context + 7-agent Byzantine consensus
```

### Root Cause Detection
```bash
# Step 3: Root Cause Detection
# Graph analysis + Connascence + Raft consensus + 5-Whys
```

### Validation
```bash
# Root causes validated by:
# - Byzantine consensus (5/7 agents agree)
# - Raft consensus (majority of graph analysts agree)
# - 5-Whys methodology (reveals true root, not symptom)
# - Connascence analysis (determines fix strategy)
```

## Best Practices

### 1. Always Use Multiple Techniques
- 5-Whys for depth
- Graph analysis for relationships
- Connascence for fix strategy
- Consensus for validation

### 2. Document Reasoning
- Capture full 5-Whys chain
- Store graph analysis results
- Record agent votes and disagreements

### 3. Validate with Consensus
- Byzantine consensus (7 agents, 5/7 required)
- Raft consensus (leader-based validation)
- Never trust single-agent analysis

### 4. Feed Back to Loop 1
- Extract failure patterns
- Generate pre-mortem questions
- Provide prevention strategies

### 5. Avoid Common Mistakes
- Don't stop at symptoms
- Don't skip consensus validation
- Don't ignore connascence (atomic fix requirements)
- Don't treat all failures as independent (check for cascades)

## Related Documentation

- [Recovery Strategies](./recovery-strategies.md) - Automated repair patterns
- [Example 1: Test Failure Recovery](../examples/example-1-test-failure-recovery.md) - Complete analysis walkthrough
- [Example 2: Build Failure Recovery](../examples/example-2-build-failure-recovery.md) - TypeScript error analysis
