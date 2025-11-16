# Implementation & Testing Plan: Memory MCP + Connascence Analyzer

**Status**: Loop 2 Initialization - Immediate Implementation
**Created**: 2025-11-01
**Goal**: Deploy both MCP servers, test all components, validate full integration

---

## Executive Summary

This plan executes **Phase 1 (Memory MCP)** and **Phase 2 (Connascence)** from the 10-week roadmap, focusing on:
1. Installing dependencies and starting both MCP servers
2. Testing every component systematically
3. Validating integration with Claude Desktop
4. Preparing for Phase 3 (Learning Loop)

**Timeline**: 2-4 weeks for complete testing and validation

---

## Phase 1: Memory MCP Implementation & Testing

### P1-A: Install Dependencies (4 hours)

**Objective**: Install all Memory MCP dependencies with Windows compatibility

**Tasks**:
1. Navigate to Memory MCP directory
2. Create Python virtual environment
3. Install core dependencies:
   - Python 3.12+
   - FastAPI
   - uvicorn
   - ChromaDB
   - hnswlib (with Windows pre-compiled binaries)
   - python-dotenv
   - pydantic
4. Install Obsidian integration:
   - markdown
   - frontmatter
   - watchdog
5. Install testing dependencies:
   - pytest
   - pytest-asyncio
   - pytest-cov

**Windows Compatibility Fix**:
```bash
# Use pre-compiled binaries for hnswlib
pip install hnswlib --only-binary :all:

# If fails, use WSL2 Docker fallback
```

**Success Criteria**:
- ✅ All packages installed without errors
- ✅ `pip list` shows all dependencies
- ✅ No native binding errors

**Validation Command**:
```bash
cd C:\Users\17175\Desktop\memory-mcp-triple-system
python -c "import chromadb; import hnswlib; print('All imports successful')"
```

---

### P1-B: Start Memory MCP Server (2 hours)

**Objective**: Launch MCP server and verify health

**Tasks**:
1. Create `.env` configuration file
2. Start server: `python -m src.mcp.server`
3. Verify health endpoint: `curl http://127.0.0.1:8000/health`
4. Check server logs for errors
5. Verify stdio transport mode

**Environment Configuration** (`.env`):
```env
MCP_SERVER_HOST=127.0.0.1
MCP_SERVER_PORT=8000
MCP_TRANSPORT=stdio
SHORT_TERM_RETENTION=86400
MID_TERM_RETENTION=604800
LONG_TERM_RETENTION=2592000
OBSIDIAN_VAULT_PATH=C:\Users\17175\Obsidian\12FA-Memory
OBSIDIAN_SYNC_ENABLED=true
OBSIDIAN_READ_ONLY=true
CHROMA_DB_PATH=./data/chroma
LOG_LEVEL=INFO
```

**Success Criteria**:
- ✅ Server starts without errors
- ✅ Health endpoint returns `{"status": "healthy"}`
- ✅ No port conflicts
- ✅ Logs show successful initialization

---

### P1-C: Test Memory MCP Basic Operations (4 hours)

**Objective**: Validate core memory operations (store, retrieve, query)

**Test Cases**:

#### Test 1: Store & Retrieve
```python
# test_basic_operations.py
import requests

# Store entry
response = requests.post('http://127.0.0.1:8000/memory/store', json={
    'key': 'test-1',
    'value': 'Hello from Memory MCP',
    'namespace': 'testing',
    'retention_policy': 'short-term'
})
assert response.status_code == 200

# Retrieve entry
response = requests.get('http://127.0.0.1:8000/memory/retrieve', params={
    'key': 'test-1',
    'namespace': 'testing'
})
assert response.json()['value'] == 'Hello from Memory MCP'
```

#### Test 2: Semantic Search
```python
# Store multiple entries
entries = [
    {'key': 'doc-1', 'value': 'Python programming tutorial', 'namespace': 'docs'},
    {'key': 'doc-2', 'value': 'JavaScript async patterns', 'namespace': 'docs'},
    {'key': 'doc-3', 'value': 'Python best practices', 'namespace': 'docs'}
]
for entry in entries:
    requests.post('http://127.0.0.1:8000/memory/store', json=entry)

# Query by semantic similarity
response = requests.post('http://127.0.0.1:8000/memory/query', json={
    'query': 'python coding',
    'namespace': 'docs',
    'limit': 2
})
results = response.json()
assert len(results) == 2
assert 'Python' in results[0]['value']
```

#### Test 3: Namespace Isolation
```python
# Store in different namespaces
requests.post('http://127.0.0.1:8000/memory/store', json={
    'key': 'key-1', 'value': 'Namespace A', 'namespace': 'ns-a'
})
requests.post('http://127.0.0.1:8000/memory/store', json={
    'key': 'key-1', 'value': 'Namespace B', 'namespace': 'ns-b'
})

# Retrieve from ns-a
response = requests.get('http://127.0.0.1:8000/memory/retrieve', params={
    'key': 'key-1', 'namespace': 'ns-a'
})
assert response.json()['value'] == 'Namespace A'

# Retrieve from ns-b
response = requests.get('http://127.0.0.1:8000/memory/retrieve', params={
    'key': 'key-1', 'namespace': 'ns-b'
})
assert response.json()['value'] == 'Namespace B'
```

**Success Criteria**:
- ✅ All 3 test cases pass
- ✅ Store operation completes <100ms
- ✅ Retrieve operation completes <50ms
- ✅ Query operation completes <200ms

---

### P1-D: Test 3-Layer Retention (8 hours)

**Objective**: Validate memory promotion (short → mid → long)

**Test Strategy**: Use time manipulation for fast testing

#### Test 1: Short-Term Retention (24h)
```python
# Store with short-term policy
response = requests.post('http://127.0.0.1:8000/memory/store', json={
    'key': 'short-test',
    'value': 'Short-term data',
    'namespace': 'retention-test',
    'retention_policy': 'short-term',
    'score': 60  # Above mid-term threshold (50%)
})

# Simulate 24h passage (mock time advancement)
requests.post('http://127.0.0.1:8000/admin/advance-time', json={'hours': 24})

# Verify promotion to mid-term
response = requests.get('http://127.0.0.1:8000/memory/get-retention-layer', params={
    'key': 'short-test',
    'namespace': 'retention-test'
})
assert response.json()['layer'] == 'mid-term'
```

#### Test 2: Mid-Term Retention (7d)
```python
# Store with mid-term score
response = requests.post('http://127.0.0.1:8000/memory/store', json={
    'key': 'mid-test',
    'value': 'Mid-term data',
    'namespace': 'retention-test',
    'retention_policy': 'mid-term',
    'score': 15  # Above long-term threshold (10%)
})

# Simulate 7d passage
requests.post('http://127.0.0.1:8000/admin/advance-time', json={'days': 7})

# Verify promotion to long-term
response = requests.get('http://127.0.0.1:8000/memory/get-retention-layer', params={
    'key': 'mid-test',
    'namespace': 'retention-test'
})
assert response.json()['layer'] == 'long-term'
```

#### Test 3: Expiration
```python
# Store with low score (won't promote)
response = requests.post('http://127.0.0.1:8000/memory/store', json={
    'key': 'expire-test',
    'value': 'Low-score data',
    'namespace': 'retention-test',
    'retention_policy': 'short-term',
    'score': 5  # Below mid-term threshold
})

# Simulate 24h passage
requests.post('http://127.0.0.1:8000/admin/advance-time', json={'hours': 24})

# Verify entry expired (deleted)
response = requests.get('http://127.0.0.1:8000/memory/retrieve', params={
    'key': 'expire-test',
    'namespace': 'retention-test'
})
assert response.status_code == 404  # Not found
```

**Success Criteria**:
- ✅ Short-term → Mid-term promotion works (score ≥50%)
- ✅ Mid-term → Long-term promotion works (score ≥10%)
- ✅ Low-score entries expire correctly
- ✅ Retention policies enforced accurately

---

### P1-E: Test Obsidian Vault Sync (6 hours)

**Objective**: Validate bidirectional sync with Obsidian vault

#### Setup: Create Obsidian Vault
```bash
mkdir -p C:\Users\17175\Obsidian\12FA-Memory\{short-term,mid-term,long-term,patterns}
```

#### Test 1: Memory → Obsidian (Read-Only Sync)
```python
# Store entry
response = requests.post('http://127.0.0.1:8000/memory/store', json={
    'key': 'obsidian-test-1',
    'value': 'Test data for Obsidian sync',
    'namespace': 'testing',
    'retention_policy': 'short-term'
})

# Wait for sync (5 min interval)
time.sleep(300)

# Check Obsidian vault file created
import os
vault_path = 'C:\\Users\\17175\\Obsidian\\12FA-Memory\\short-term\\testing\\obsidian-test-1.md'
assert os.path.exists(vault_path)

# Read file content
with open(vault_path, 'r') as f:
    content = f.read()
    assert 'Test data for Obsidian sync' in content
    assert '---' in content  # Frontmatter present
```

#### Test 2: Frontmatter Validation
```python
import frontmatter

vault_file = 'C:\\Users\\17175\\Obsidian\\12FA-Memory\\short-term\\testing\\obsidian-test-1.md'
with open(vault_file, 'r') as f:
    post = frontmatter.load(f)

    # Validate frontmatter
    assert post['key'] == 'obsidian-test-1'
    assert post['namespace'] == 'testing'
    assert post['retention'] == 'short-term'
    assert 'created' in post

    # Validate content
    assert post.content == 'Test data for Obsidian sync'
```

#### Test 3: File Watcher (if bidirectional enabled)
```python
# NOTE: Only if OBSIDIAN_READ_ONLY=false

# Manually edit vault file
vault_file = 'C:\\Users\\17175\\Obsidian\\12FA-Memory\\short-term\\testing\\obsidian-test-1.md'
with open(vault_file, 'w') as f:
    f.write("""---
key: obsidian-test-1
namespace: testing
retention: short-term
---
EDITED: Manual edit from Obsidian
""")

# Wait for file watcher to detect change
time.sleep(10)

# Verify Memory MCP updated
response = requests.get('http://127.0.0.1:8000/memory/retrieve', params={
    'key': 'obsidian-test-1',
    'namespace': 'testing'
})
assert 'EDITED: Manual edit from Obsidian' in response.json()['value']
```

**Success Criteria**:
- ✅ Memory entries create markdown files in vault
- ✅ Frontmatter contains all metadata
- ✅ Sync latency <5 minutes
- ✅ File watcher detects changes (if bidirectional)

---

## Phase 2: Connascence Analyzer Implementation & Testing

### P2-A: Install Dependencies (2 hours)

**Objective**: Install Connascence analyzer dependencies

**Tasks**:
1. Navigate to Connascence directory
2. Create Python virtual environment (separate from Memory MCP)
3. Install core dependencies:
   - Python 3.12+
   - FastAPI
   - uvicorn
   - AST parsing libraries
   - networkx
   - graphviz
4. Install testing dependencies:
   - pytest
   - pytest-asyncio

**Success Criteria**:
- ✅ All packages installed
- ✅ No dependency conflicts with Memory MCP
- ✅ Separate virtual environment working

---

### P2-B: Start Connascence MCP Server (2 hours)

**Objective**: Launch Connascence server on different port

**Environment Configuration** (`.env`):
```env
MCP_SERVER_HOST=127.0.0.1
MCP_SERVER_PORT=8001  # Different from Memory MCP (8000)
MCP_TRANSPORT=stdio
SUPPORTED_LANGUAGES=javascript,typescript
ANALYSIS_PROFILE=nasa-compliance
INCREMENTAL_CACHE_ENABLED=true
CACHE_PATH=./data/cache
DEFAULT_WORKSPACE=C:\Users\17175
EXCLUDE_PATTERNS=node_modules,dist,build,.git,venv
LOG_LEVEL=INFO
```

**Success Criteria**:
- ✅ Server starts on port 8001
- ✅ Health endpoint returns `{"status": "healthy"}`
- ✅ No port conflicts with Memory MCP

---

### P2-C: Test All 9 Connascence Types (6 hours)

**Objective**: Validate detection of all 9 connascence types

**Test File**: Create `test-connascence-types.js` with known violations

#### Test 1: CoN (Connascence of Name)
```javascript
// test-files/CoN-test.js
function calculateTotal(price, quantity) {
  return price * quantity;  // CoN: 'price', 'quantity' names must match
}

const result = calculateTotal(10, 5);  // Must use same parameter names
```

**Expected Output**:
```json
{
  "type": "CoN",
  "severity": "low",
  "location": "line 2",
  "strength": 1,
  "locality": 1,
  "degree": 2,
  "message": "Name coupling detected: parameters 'price', 'quantity'"
}
```

#### Test 2: CoT (Connascence of Type)
```javascript
// test-files/CoT-test.js
function processData(data) {
  return data.map(x => x.value);  // CoT: 'data' must be Array type
}

const result = processData([{value: 1}, {value: 2}]);  // Must pass Array
```

**Expected Output**:
```json
{
  "type": "CoT",
  "severity": "medium",
  "strength": 2,
  "message": "Type coupling: 'data' must be Array"
}
```

#### Test 3: CoM (Connascence of Meaning)
```javascript
// test-files/CoM-test.js
function setStatus(status) {
  // CoM: 1=active, 2=inactive, 3=pending (magic numbers)
  if (status === 1) return 'Active';
  if (status === 2) return 'Inactive';
  if (status === 3) return 'Pending';
}

setStatus(1);  // Caller must know 1 = active
```

**Expected Output**:
```json
{
  "type": "CoM",
  "severity": "high",
  "strength": 3,
  "message": "Meaning coupling: magic number '1' has implicit meaning"
}
```

#### Test 4: CoP (Connascence of Position)
```javascript
// test-files/CoP-test.js
function createUser(name, email, age, city) {
  // CoP: parameter order matters
  return { name, email, age, city };
}

createUser('John', 'john@example.com', 30, 'NYC');  // Order must match
```

**Expected Output**:
```json
{
  "type": "CoP",
  "severity": "medium",
  "strength": 2,
  "degree": 4,
  "message": "Position coupling: 4 parameters in fixed order"
}
```

#### Test 5: CoA (Connascence of Algorithm)
```javascript
// test-files/CoA-test.js
function hashPassword(password) {
  // CoA: Both sides must use same hashing algorithm
  return btoa(password);  // Base64 encoding
}

function verifyPassword(input, hash) {
  return btoa(input) === hash;  // Must use same algorithm
}
```

**Expected Output**:
```json
{
  "type": "CoA",
  "severity": "high",
  "strength": 4,
  "message": "Algorithm coupling: hash/verify must use same algorithm"
}
```

#### Test 6: CoE (Connascence of Execution)
```javascript
// test-files/CoE-test.js
let globalState = 0;

function increment() {
  globalState++;  // CoE: Must execute before read
}

function getValue() {
  return globalState;  // CoE: Depends on increment() being called first
}

increment();  // MUST execute before getValue()
const value = getValue();
```

**Expected Output**:
```json
{
  "type": "CoE",
  "severity": "critical",
  "strength": 5,
  "message": "Execution order coupling: increment() must run before getValue()"
}
```

#### Test 7: CoV (Connascence of Value)
```javascript
// test-files/CoV-test.js
const MAX_RETRIES = 3;

function retry(fn) {
  for (let i = 0; i < MAX_RETRIES; i++) {  // CoV: both use MAX_RETRIES value
    try {
      return fn();
    } catch (e) {}
  }
}

function checkRetries() {
  return MAX_RETRIES;  // CoV: value must match
}
```

**Expected Output**:
```json
{
  "type": "CoV",
  "severity": "low",
  "strength": 1,
  "message": "Value coupling: MAX_RETRIES shared across functions"
}
```

#### Test 8: CoI (Connascence of Identity)
```javascript
// test-files/CoI-test.js
const sharedConfig = { timeout: 5000 };

function setConfig(config) {
  config.timeout = 10000;  // CoI: Mutates shared object
}

function getConfig() {
  return sharedConfig;  // CoI: Same object identity
}

setConfig(sharedConfig);  // Modifies shared config
console.log(getConfig().timeout);  // 10000 (mutated)
```

**Expected Output**:
```json
{
  "type": "CoI",
  "severity": "high",
  "strength": 4,
  "message": "Identity coupling: shared object mutation"
}
```

#### Test 9: CoId (Connascence of Identity - Reference)
```javascript
// test-files/CoId-test.js
let singleton = null;

function getInstance() {
  if (!singleton) {
    singleton = { value: 42 };
  }
  return singleton;  // CoId: Always returns same reference
}

const a = getInstance();
const b = getInstance();
console.log(a === b);  // true (same reference)
```

**Expected Output**:
```json
{
  "type": "CoId",
  "severity": "medium",
  "strength": 3,
  "message": "Identity reference coupling: singleton pattern"
}
```

**Test Execution**:
```bash
cd C:\Users\17175\Desktop\connascence

# Analyze each test file
python -m src.cli analyze test-files/CoN-test.js
python -m src.cli analyze test-files/CoT-test.js
python -m src.cli analyze test-files/CoM-test.js
# ... (all 9 types)

# Or analyze all at once
python -m src.cli analyze test-files/ --profile nasa-compliance
```

**Success Criteria**:
- ✅ All 9 connascence types detected
- ✅ 0% false positives (verified against known violations)
- ✅ Strength, locality, degree calculated correctly
- ✅ Analysis completes <5s per file

---

### P2-D: Test Workspace Analysis (4 hours)

**Objective**: Validate workspace-wide analysis with caching

#### Test 1: Full Workspace Analysis
```bash
# Create test workspace with 50+ files
mkdir -p C:\Users\17175\test-workspace\src
# Copy real code files or create test files

# Run workspace analysis (first time, cold cache)
time python -m src.cli analyze-workspace C:\Users\17175\test-workspace

# Expected: <15s for 50 files
```

#### Test 2: Incremental Analysis (Cache)
```bash
# Edit one file
echo "function test() {}" >> C:\Users\17175\test-workspace\src\edited.js

# Run analysis again (warm cache)
time python -m src.cli analyze-workspace C:\Users\17175\test-workspace --incremental

# Expected: <5s (only re-analyzes edited.js)
```

#### Test 3: Exclude Patterns
```bash
# Create node_modules directory (should be excluded)
mkdir -p C:\Users\17175\test-workspace\node_modules
echo "fake package" > C:\Users\17175\test-workspace\node_modules\fake.js

# Run analysis
python -m src.cli analyze-workspace C:\Users\17175\test-workspace

# Verify node_modules not analyzed
python -m src.cli get-cache-info | grep -v "node_modules"
```

**Success Criteria**:
- ✅ Full analysis <15s for 100 files
- ✅ Incremental analysis <5s
- ✅ Cache hit rate >80% on second run
- ✅ Exclude patterns working (node_modules, dist, etc.)

---

## Phase 3: Claude Desktop MCP Integration (2 hours)

**Objective**: Register both MCP servers in Claude Desktop

### Task: Update `mcp_servers.json`

**Location**: `C:\Users\17175\.config\claude\mcp_servers.json`

**Configuration**:
```json
{
  "mcpServers": {
    "memory-mcp": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "C:\\Users\\17175\\Desktop\\memory-mcp-triple-system",
      "env": {
        "PYTHONPATH": "C:\\Users\\17175\\Desktop\\memory-mcp-triple-system",
        "OBSIDIAN_VAULT_PATH": "C:\\Users\\17175\\Obsidian\\12FA-Memory"
      },
      "transport": "stdio",
      "description": "3-layer persistent memory with Obsidian vault synchronization",
      "enabled": true
    },
    "connascence-analyzer": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "C:\\Users\\17175\\Desktop\\connascence",
      "env": {
        "PYTHONPATH": "C:\\Users\\17175\\Desktop\\connascence",
        "DEFAULT_WORKSPACE": "C:\\Users\\17175"
      },
      "transport": "stdio",
      "description": "Comprehensive code coupling analysis (9 connascence types)",
      "enabled": true
    },
    "claude-flow": {
      "command": "npx",
      "args": ["claude-flow@alpha", "mcp", "start"],
      "transport": "stdio",
      "description": "claude-flow coordination and hot memory",
      "enabled": true
    }
  }
}
```

**Validation**:
1. Restart Claude Desktop
2. Check MCP server indicators in status bar
3. Verify 3 green dots: 🟢 memory-mcp, 🟢 connascence-analyzer, 🟢 claude-flow

**Success Criteria**:
- ✅ JSON syntax valid
- ✅ Both servers start automatically
- ✅ Claude Desktop detects both servers
- ✅ No startup errors in Claude Desktop logs

---

## Phase 4: End-to-End Integration Testing (4 hours)

**Objective**: Validate full workflow with both MCP systems

### Test 1: Memory MCP from Claude Code
```javascript
// In Claude Code session
await memory.store({
  key: 'e2e-test-1',
  value: 'End-to-end test data',
  namespace: 'integration-test',
  retention_policy: 'short-term'
});

const result = await memory.retrieve({
  key: 'e2e-test-1',
  namespace: 'integration-test'
});

console.assert(result === 'End-to-end test data', 'E2E test failed!');
```

### Test 2: Connascence from Claude Code
```javascript
// In Claude Code session
const analysis = await connascence.analyze_file({
  file_path: 'C:\\Users\\17175\\test-files\\CoN-test.js'
});

console.assert(analysis.violations.length > 0, 'No violations detected!');
console.assert(analysis.violations[0].type === 'CoN', 'Wrong connascence type!');
```

### Test 3: Agent-Specific Access Control
```javascript
// Spawn coder agent (HAS connascence access)
const coderResult = await Task("Test coder agent",
  "Analyze test-files/CoN-test.js for coupling issues",
  "coder");

// Spawn planner agent (NO connascence access)
const plannerResult = await Task("Test planner agent",
  "Try to analyze coupling (should fail)",
  "planner");

// Verify coder can access connascence
console.assert(coderResult.includes('CoN'), 'Coder should access connascence!');

// Verify planner blocked from connascence
console.assert(plannerResult.includes('blocked'), 'Planner should be blocked!');
```

**Success Criteria**:
- ✅ Memory MCP tools accessible from Claude Code
- ✅ Connascence tools accessible from Claude Code
- ✅ Agent access control enforced correctly
- ✅ No errors in end-to-end workflows

---

## Summary: Test Matrix

| Component | Tests | Status | Time |
|-----------|-------|--------|------|
| **Memory MCP** | | | |
| P1-A: Dependencies | Installation validation | ⏳ Pending | 4h |
| P1-B: Server startup | Health check | ⏳ Pending | 2h |
| P1-C: Basic operations | Store/Retrieve/Query | ⏳ Pending | 4h |
| P1-D: 3-layer retention | Promotion logic | ⏳ Pending | 8h |
| P1-E: Obsidian sync | Bidirectional sync | ⏳ Pending | 6h |
| **Connascence** | | | |
| P2-A: Dependencies | Installation validation | ⏳ Pending | 2h |
| P2-B: Server startup | Health check | ⏳ Pending | 2h |
| P2-C: 9 connascence types | All types detection | ⏳ Pending | 6h |
| P2-D: Workspace analysis | Incremental caching | ⏳ Pending | 4h |
| **Integration** | | | |
| P3: Claude Desktop | MCP registration | ⏳ Pending | 2h |
| P4: End-to-end | Full workflow | ⏳ Pending | 4h |
| **TOTAL** | | | **44h** |

---

## Next Steps

1. **Execute Phase 1A**: Install Memory MCP dependencies
2. **Execute Phase 1B**: Start Memory MCP server
3. **Parallel execution**: While testing Memory MCP, install Connascence dependencies
4. **Iterate**: Fix issues as they arise, document solutions
5. **Validate**: Run full test suite before proceeding to Phase 3 (Learning Loop)

---

**Status**: Ready for Loop 2 Execution
**Approval**: Pending user confirmation to proceed
