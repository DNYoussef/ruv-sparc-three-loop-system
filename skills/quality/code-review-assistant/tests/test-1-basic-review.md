# Test 1: Basic Code Review

**Test ID**: TEST-CRA-001
**Skill**: code-review-assistant
**Tier**: Gold
**Focus**: Basic multi-agent review functionality

---

## Test Objective

Validate that code-review-assistant can:
1. Initialize 5-agent swarm successfully
2. Execute parallel reviews across all agents
3. Generate comprehensive review report
4. Calculate accurate overall score
5. Determine merge readiness

---

## Test Setup

### Prerequisites
```bash
# Ensure required tools installed
which gh
which npx
which python3

# Verify MCP servers active
npx claude-flow coordination swarm-status
```

### Test Repository
Create simple test repository with known issues:

```bash
mkdir -p /tmp/test-repo-basic
cd /tmp/test-repo-basic
git init

# Create files with intentional issues
cat > auth.py <<'EOF'
import hashlib

def login(username, password):
    # Hardcoded secret (SEC-001)
    api_key = "sk-1234567890abcdef"

    # Weak crypto (SEC-102)
    hashed = hashlib.md5(password.encode()).hexdigest()

    # SQL injection vulnerability (SEC-003)
    query = "SELECT * FROM users WHERE name='" + username + "'"

    return query
EOF

cat > slow_function.py <<'EOF'
def process_data(data):
    # Nested loops - O(n²) (PERF-002)
    result = []
    for i in data:
        for j in data:
            if i == j:
                result.append(i)
    return result

# No docstring (DOC-001)
def helper():
    pass
EOF

cat > style_issues.js <<'EOF'
// Using var instead of const (STYLE-SEC-102)
var apiKey = "secret123";

// console.log in production (STYLE-debugging)
console.log(apiKey);

// No tests for this file (TEST-001)
function processUser(user) {
    return user.name;
}
EOF

# Commit files
git add .
git commit -m "Initial commit with test issues"
```

---

## Test Execution

### Step 1: Direct Skill Invocation

```bash
# Run code-review-assistant on test repo
cd /tmp/test-repo-basic

# Execute skill
code-review-assistant --pr-number 1 \
  --changed-files "auth.py,slow_function.py,style_issues.js" \
  --focus-areas "security,performance,style,tests,documentation"
```

### Step 2: Verify Agent Spawning

Expected output:
```
[MultiAgentReviewer] Initializing swarm for PR #1
[MultiAgentReviewer] Focus areas: security, performance, style, tests, documentation
[MultiAgentReviewer] Files to review: 3
[MultiAgentReviewer] Swarm initialized successfully
[MultiAgentReviewer] Spawning 5 specialized agents...
  ✓ Security Reviewer spawned
  ✓ Performance Analyst spawned
  ✓ Style Reviewer spawned
  ✓ Test Specialist spawned
  ✓ Documentation Reviewer spawned
```

### Step 3: Verify Parallel Reviews

Expected output:
```
[MultiAgentReviewer] Executing 5 reviews in parallel...
[Security Reviewer] Starting review...
[Performance Analyst] Starting review...
[Style Reviewer] Starting review...
[Test Specialist] Starting review...
[Documentation Reviewer] Starting review...
```

### Step 4: Verify Issue Detection

Expected findings:
- **Security**: 3 critical issues (hardcoded secret, weak crypto, SQL injection)
- **Performance**: 1 medium issue (nested loops)
- **Style**: 2 medium issues (var usage, console.log)
- **Tests**: 1 high issue (no tests)
- **Documentation**: 1 medium issue (missing docstring)

---

## Expected Results

### Overall Score Calculation

```
Security Score: 10/100 (3 critical × 30 = -90)
Performance Score: 90/100 (1 medium × 10 = -10)
Style Score: 90/100 (2 medium × 5 = -10)
Test Score: 0/100 (no test files)
Documentation Score: 92/100 (1 medium × 8 = -8)

Weighted Overall Score:
  (10 × 0.30) + (90 × 0.25) + (90 × 0.15) + (0 × 0.20) + (92 × 0.10)
  = 3 + 22.5 + 13.5 + 0 + 9.2
  = 48.2/100
```

### Merge Decision

```yaml
merge_ready: false
blocking_issues: 3 (critical security issues)
warnings: 4 (performance, style, tests, docs)
decision: "request_changes"
reason: "Critical security vulnerabilities must be addressed"
```

### JSON Output Structure

```json
{
  "pr_number": 1,
  "pr_title": "Test PR with known issues",
  "overall_score": 48.2,
  "merge_ready": false,
  "blocking_issues": 3,
  "warnings": 4,
  "suggestions": 0,
  "agent_reviews": [
    {
      "agent_name": "Security Reviewer",
      "score": 10,
      "findings": [
        {
          "severity": "critical",
          "category": "secrets",
          "message": "Hardcoded API key",
          "file": "auth.py",
          "line": 5,
          "suggestion": "Use environment variables"
        }
      ]
    }
  ],
  "timestamp": "2025-11-02T...",
  "execution_time": 2.45
}
```

---

## Validation Criteria

### ✅ Pass Conditions

1. **Swarm Initialization**
   - All 5 agents spawn successfully
   - Mesh topology configured
   - No initialization errors

2. **Issue Detection**
   - All 8 intentional issues detected
   - Correct severity assignments
   - Accurate file and line numbers

3. **Score Calculation**
   - Overall score ≈ 48 ± 5 points
   - Individual agent scores accurate
   - Weighted calculation correct

4. **Merge Decision**
   - `merge_ready = false`
   - Blocking issues identified
   - Decision reasoning clear

5. **Performance**
   - Total execution time < 5 seconds
   - Parallel execution confirmed
   - JSON output valid

### ❌ Fail Conditions

- Any agent fails to spawn
- Critical issues not detected
- Score calculation incorrect (>10 point error)
- Merge decision contradicts findings
- Execution time > 10 seconds

---

## Cleanup

```bash
# Remove test repository
rm -rf /tmp/test-repo-basic

# Verify cleanup
ls /tmp/test-repo-basic 2>&1 | grep "No such file"
```

---

## Test Results

**Date**: _____________
**Executor**: _____________
**Status**: [ ] PASS | [ ] FAIL | [ ] SKIP

**Notes**:
-
-
-

**Issues Found**:
-
-
