# Test 3: Root Cause Analysis with Graph-Based Cascade Detection

**Skill**: cicd-intelligent-recovery
**Phase**: Step 3 (Root Cause Detection)
**Version**: 2.0.0

## Test Objective

Validate that the root cause analysis correctly:
1. Builds failure dependency graphs using multiple heuristics
2. Identifies root causes (failures with no incoming dependencies)
3. Detects cascade failures (failures caused by root failures)
4. Applies 5-Whys methodology for validation
5. Generates Raft consensus on final root cause list
6. Calculates cascade depth and impact

## Prerequisites

```bash
# Create test directory
mkdir -p .claude/.artifacts/test-rootcause

# Create mock parsed failures with dependencies
cat > .claude/.artifacts/test-rootcause/parsed-failures.json << 'EOF'
[
  {
    "testName": "should initialize database pool",
    "file": "src/database/pool.js",
    "line": 25,
    "column": 10,
    "errorMessage": "Error: Connection refused on localhost:5432",
    "runId": "run-001"
  },
  {
    "testName": "should create user table",
    "file": "src/database/migrations/001_users.js",
    "line": 15,
    "column": 5,
    "errorMessage": "Error: Cannot execute query - no database connection from pool.js",
    "runId": "run-001"
  },
  {
    "testName": "should save user to database",
    "file": "src/models/user.js",
    "line": 42,
    "column": 8,
    "errorMessage": "Error: User table does not exist - migration from 001_users.js failed",
    "runId": "run-001"
  },
  {
    "testName": "should authenticate user",
    "file": "src/auth/login.js",
    "line": 67,
    "column": 12,
    "errorMessage": "Error: Cannot find user - user.js save failed",
    "runId": "run-001"
  },
  {
    "testName": "should validate password",
    "file": "src/auth/validators.js",
    "line": 22,
    "column": 15,
    "errorMessage": "TypeError: Cannot read property 'length' of null",
    "runId": "run-001"
  }
]
EOF

# Create mock Gemini analysis with dependency graph
cat > .claude/.artifacts/test-rootcause/gemini-analysis.json << 'EOF'
{
  "dependency_graph": {
    "nodes": [
      {"id": "src/database/pool.js", "type": "module"},
      {"id": "src/database/migrations/001_users.js", "type": "migration"},
      {"id": "src/models/user.js", "type": "model"},
      {"id": "src/auth/login.js", "type": "service"}
    ],
    "edges": [
      {"from": "src/database/pool.js", "to": "src/database/migrations/001_users.js"},
      {"from": "src/database/migrations/001_users.js", "to": "src/models/user.js"},
      {"from": "src/models/user.js", "to": "src/auth/login.js"}
    ]
  }
}
EOF
```

## Test Execution

### Step 1: Run Root Cause Analysis

```bash
# Execute root cause analyzer
cd .claude/.artifacts/test-rootcause
python3 ../../../resources/scripts/root_cause.py .
```

**Expected Output**:
```
=== Root Cause Analysis ===

1. Loading failures...
   Loaded 5 failures
2. Loading Gemini context...
3. Building failure dependency graph...
   Nodes: 5
   Edges: 4
4. Identifying root causes...
   Root failures: 2
5. Detecting circular dependencies...
   ✅ No circular dependencies
6. Validating root causes with 5-Whys...
7. Generating Raft consensus...

✅ Root cause analysis complete
   Root causes: 2
   Cascaded failures: 3
   Cascade ratio: 60.00%
   Saved to: ./root-causes-consensus.json
```

### Step 2: Validate Graph Construction

```bash
# Check failure dependency graph
cat root-causes-consensus.json | jq '.roots[] | {rootId, cascadedCount: (.cascadedFailures | length)}'
```

**Expected**:
```json
{
  "rootId": "src/database/pool.js:25:should initialize database pool",
  "cascadedCount": 3
}
{
  "rootId": "src/auth/validators.js:22:should validate password",
  "cascadedCount": 0
}
```

**Validation**:
- ✅ 2 root causes identified
- ✅ Database pool failure has 3 cascaded failures
- ✅ Validators failure is isolated (no cascades)

### Step 3: Validate Cascade Chain

```bash
# Check cascade chain for database root cause
cat root-causes-consensus.json | jq '.roots[] | select(.rootId | contains("pool.js")) | .cascadedFailures'
```

**Expected**:
```json
[
  "src/database/migrations/001_users.js:15:should create user table",
  "src/models/user.js:42:should save user to database",
  "src/auth/login.js:67:should authenticate user"
]
```

**Validation**:
- ✅ Cascade follows dependency chain: pool → migration → model → auth
- ✅ All 3 downstream failures are identified

### Step 4: Validate Fix Strategy Assignment

```bash
# Check fix strategies
cat root-causes-consensus.json | jq '.roots[] | {rootId, fixStrategy, fixComplexity}'
```

**Expected**:
```json
{
  "rootId": "src/database/pool.js:25:should initialize database pool",
  "fixStrategy": "bundled",
  "fixComplexity": "moderate"
}
{
  "rootId": "src/auth/validators.js:22:should validate password",
  "fixStrategy": "isolated",
  "fixComplexity": "simple"
}
```

**Validation**:
- ✅ Database issue gets "bundled" strategy (has cascades)
- ✅ Validators issue gets "isolated" strategy (no cascades)
- ✅ Complexity matches cascade count (3 = moderate, 0 = simple)

### Step 5: Validate Failure Categorization

```bash
# Check failure categories
cat root-causes-consensus.json | jq '.roots[] | {rootId, category}'
```

**Expected**:
```json
{
  "rootId": "src/database/pool.js:25:should initialize database pool",
  "category": "network-resilience"
}
{
  "rootId": "src/auth/validators.js:22:should validate password",
  "category": "null-safety"
}
```

**Validation**:
- ✅ Connection refused → network-resilience
- ✅ Cannot read property of null → null-safety

### Step 6: Validate Statistics

```bash
# Check cascade statistics
cat root-causes-consensus.json | jq '.stats'
```

**Expected**:
```json
{
  "totalFailures": 5,
  "rootFailures": 2,
  "cascadedFailures": 3,
  "cascadeRatio": 0.6
}
```

**Validation**:
- ✅ Total = 5 failures
- ✅ Roots = 2 (pool, validators)
- ✅ Cascaded = 3 (migration, model, auth)
- ✅ Ratio = 60% (3/5)

## Advanced Tests

### Test 7: Circular Dependency Detection

```bash
# Add circular dependency
cat > parsed-failures-circular.json << 'EOF'
[
  {
    "testName": "service A",
    "file": "src/services/a.js",
    "line": 10,
    "errorMessage": "Error: Service B failed from b.js"
  },
  {
    "testName": "service B",
    "file": "src/services/b.js",
    "line": 20,
    "errorMessage": "Error: Service C failed from c.js"
  },
  {
    "testName": "service C",
    "file": "src/services/c.js",
    "line": 30,
    "errorMessage": "Error: Service A failed from a.js"
  }
]
EOF

# Run analysis
cp parsed-failures-circular.json parsed-failures.json
python3 ../../../resources/scripts/root_cause.py .
```

**Expected Output**:
```
5. Detecting circular dependencies...
   ⚠️  Found 1 circular dependencies
```

**Expected in Output**:
- Cycle detected: a.js → b.js → c.js → a.js
- All 3 marked as roots (circular dependencies have no clear root)

### Test 8: High Cascade Depth

```bash
# Create deep cascade (7 levels)
cat > parsed-failures-deep.json << 'EOF'
[
  {"testName": "level0", "file": "level0.js", "line": 1, "errorMessage": "Root failure"},
  {"testName": "level1", "file": "level1.js", "line": 1, "errorMessage": "Caused by level0.js"},
  {"testName": "level2", "file": "level2.js", "line": 1, "errorMessage": "Caused by level1.js"},
  {"testName": "level3", "file": "level3.js", "line": 1, "errorMessage": "Caused by level2.js"},
  {"testName": "level4", "file": "level4.js", "line": 1, "errorMessage": "Caused by level3.js"},
  {"testName": "level5", "file": "level5.js", "line": 1, "errorMessage": "Caused by level4.js"},
  {"testName": "level6", "file": "level6.js", "line": 1, "errorMessage": "Caused by level5.js"}
]
EOF

cp parsed-failures-deep.json parsed-failures.json
python3 ../../../resources/scripts/root_cause.py .

# Check cascade depth
cat root-causes-consensus.json | jq '.roots[] | .cascadeDepth'
```

**Expected**: `6` (depth from root to deepest cascade)

**Fix Strategy**: `"architectural"` (cascade count = 6 triggers architectural refactor)

### Test 9: Multiple Independent Roots

```bash
# Create multiple independent failure chains
cat > parsed-failures-multi.json << 'EOF'
[
  {"testName": "auth root", "file": "auth.js", "line": 1, "errorMessage": "Auth failed"},
  {"testName": "auth cascade", "file": "login.js", "line": 1, "errorMessage": "Auth failed from auth.js"},
  {"testName": "db root", "file": "database.js", "line": 1, "errorMessage": "DB failed"},
  {"testName": "db cascade", "file": "query.js", "line": 1, "errorMessage": "DB failed from database.js"}
]
EOF

cp parsed-failures-multi.json parsed-failures.json
python3 ../../../resources/scripts/root_cause.py .

# Count roots
cat root-causes-consensus.json | jq '.stats.rootFailures'
```

**Expected**: `2` (auth.js and database.js)

**Cascade Count**: `2` (login.js and query.js)

## Validation Criteria

✅ **PASS** if:
- Root causes are correctly identified (failures with no incoming dependencies)
- Cascade chains are accurate (following dependency graph)
- Fix strategies match cascade count (0=isolated, 1-3=bundled, 4+=architectural)
- Failure categories are correct (based on error messages)
- Circular dependencies are detected
- Statistics are accurate (total, roots, cascaded, ratio)
- Deep cascades (6+ levels) trigger architectural strategy

❌ **FAIL** if:
- Root causes are misidentified
- Cascade chains are incomplete or incorrect
- Fix strategies don't match cascade complexity
- Categories are wrong
- Circular dependencies aren't detected
- Statistics don't add up

## Performance Expectations

- Graph construction: < 100ms for 100 failures
- Root detection: < 50ms
- Cascade depth calculation: < 100ms
- Total analysis time: < 1 second for 100 failures

## Cleanup

```bash
cd ../../..
rm -rf .claude/.artifacts/test-rootcause
```

## Notes

- Graph construction uses 3 heuristics: temporal, error references, Gemini dependencies
- Raft consensus ensures validated root causes (not just symptoms)
- 5-Whys methodology prevents stopping at symptoms
- Cascade ratio indicates systemic vs. isolated failures (>50% = systemic issues)
- Architectural strategy prevents band-aid fixes for high-coupling systems

## Next Steps

After passing this test:
1. Combine with **test-2-auto-repair.md** output for complete fix pipeline
2. Run full integration test via **recovery_pipeline.sh**
3. Validate end-to-end Loop 3 workflow
