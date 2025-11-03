# Test 3: 5-Agent Swarm Coordination

**Test ID**: TEST-CRA-003
**Skill**: code-review-assistant
**Tier**: Gold
**Focus**: Full 5-agent parallel swarm review with comprehensive findings

---

## Test Objective

Validate that code-review-assistant can:
1. Spawn all 5 specialized agents concurrently
2. Execute parallel reviews without conflicts
3. Aggregate findings from multiple agents
4. Handle cross-cutting concerns (e.g., security + performance)
5. Generate unified comprehensive report
6. Demonstrate coordination benefits over sequential review

---

## Test Setup

### Complex Multi-Domain Test Repository

```bash
mkdir -p /tmp/test-repo-swarm
cd /tmp/test-repo-swarm
git init

# Create multi-domain code with overlapping concerns
mkdir -p src tests docs

# 1. File with Security + Performance + Style issues
cat > src/user_service.py <<'EOF'
import hashlib
import random

# Missing docstring (DOC-001)
class UserService:
    def __init__(self):
        # Hardcoded secret (SEC-001)
        self.api_key = "sk-1234567890"
        self.users = []

    # No docstring (DOC-001)
    def authenticate(self, username, password):
        # SQL injection (SEC-003)
        query = "SELECT * FROM users WHERE username='" + username + "'"

        # Weak crypto (SEC-102)
        hashed = hashlib.md5(password.encode()).hexdigest()

        # N+1 query pattern (PERF-001)
        for user in self.users:
            permissions = self.get_permissions(user.id)  # DB call in loop
            user.permissions = permissions

        return hashed

    def get_permissions(self, user_id):
        # Mocked DB call
        return []

    # Nested loops - O(n²) (PERF-002)
    def find_duplicates(self,data):  # Style: missing space (STYLE-002)
        result=[]  # Style: no spaces around = (STYLE-002)
        for i in data:
            for j in data:
                if i==j:  # Style: no spaces (STYLE-002)
                    result.append(i)
        return result

    def generate_token(self):
        # Insecure random (SEC-high)
        return str(random.random())
EOF

# 2. File with Tests issues + Documentation gaps
cat > src/payment.js <<'EOF'
// No tests for this critical module (TEST-001)
// No JSDoc documentation (DOC-001)

var paymentApi = "https://api.stripe.com";  // var usage (STYLE-best-practices)

function processPayment(amount, cardNumber) {
  // No input validation (SEC-medium)
  // console.log in production (STYLE-debugging)
  console.log("Processing payment:", cardNumber);

  // Blocking sync operation (PERF-blocking)
  const result = syncHttpCall(paymentApi);

  // No error handling (STYLE-anti-pattern)
  return result;
}

function syncHttpCall(url) {
  // Simulated sync call - blocks event loop
  var start = Date.now();
  while (Date.now() - start < 1000) {}  // 1 second blocking
  return { status: "ok" };
}

// Missing event listener cleanup (PERF-memory-leak)
document.addEventListener('click', function(e) {
  // Never removed
});

// Comparison with true (STYLE-best-practices)
function isActive(user) {
  if (user.active == true) {
    return true;
  }
  return false;
}
EOF

# 3. Test file with quality issues
cat > tests/test_user_service.py <<'EOF'
# Test file with poor quality (TEST-002)

def test_authenticate():
    # No setup/teardown (TEST-101)
    # No assertions! (TEST-103)
    user_service = UserService()
    result = user_service.authenticate("admin", "password")
    # Missing assertion - test does nothing

# Bare except clause (STYLE-best-practices)
def test_find_duplicates():
    try:
        service = UserService()
        service.find_duplicates([1, 2, 3])
    except:
        pass  # Empty catch (STYLE-anti-pattern)
EOF

# 4. Documentation gaps
cat > README.md <<'EOF'
# User Management System

This is a user management system.

## Installation
Install it.

## Usage
Use it.
EOF

cat > docs/API.md <<'EOF'
# API Documentation

Coming soon...
EOF

git add .
git commit -m "Multi-domain test code for swarm review"
```

---

## Test Execution

### Step 1: Initialize Full Swarm

```bash
cd /tmp/test-repo-swarm

# Run full 5-agent swarm review
code-review-assistant --pr-number 3 \
  --changed-files "src/user_service.py,src/payment.js,tests/test_user_service.py,README.md,docs/API.md" \
  --focus-areas "security,performance,style,tests,documentation" \
  --suggest-fixes true
```

### Step 2: Verify Parallel Execution

Monitor logs for concurrent agent execution:
```
[MultiAgentReviewer] Spawning 5 specialized agents...
  ✓ Security Reviewer spawned
  ✓ Performance Analyst spawned
  ✓ Style Reviewer spawned
  ✓ Test Specialist spawned
  ✓ Documentation Reviewer spawned

[MultiAgentReviewer] Executing 5 reviews in parallel...
[Security Reviewer] Starting review...          [Thread 1]
[Performance Analyst] Starting review...        [Thread 2]
[Style Reviewer] Starting review...             [Thread 3]
[Test Specialist] Starting review...            [Thread 4]
[Documentation Reviewer] Starting review...     [Thread 5]

[Security Reviewer] Complete - Score: 25/100 (8 findings)
[Performance Analyst] Complete - Score: 75/100 (3 findings)
[Style Reviewer] Complete - Score: 80/100 (7 findings)
[Test Specialist] Complete - Score: 40/100 (4 findings)
[Documentation Reviewer] Complete - Score: 60/100 (5 findings)
```

---

## Expected Cross-Agent Findings

### Security Reviewer (Priority 1)

**Critical Issues (3)**:
1. `user_service.py:11` - Hardcoded API key
2. `user_service.py:17` - SQL injection vulnerability
3. `user_service.py:20` - Weak MD5 hashing

**High Issues (2)**:
4. `user_service.py:33` - Insecure random for tokens
5. `payment.js:8` - No input validation

**Score**: 25/100

### Performance Analyst (Priority 2)

**High Issues (1)**:
1. `user_service.py:23` - N+1 query pattern

**Medium Issues (2)**:
2. `user_service.py:30` - Nested loops O(n²)
3. `payment.js:12` - Blocking synchronous operation

**Low Issues (1)**:
4. `payment.js:23` - Event listener without cleanup

**Score**: 75/100

### Style Reviewer (Priority 3)

**Medium Issues (4)**:
1. `user_service.py:29` - Missing space after comma
2. `user_service.py:30` - No spaces around =
3. `user_service.py:32` - No spaces around ==
4. `payment.js:5` - Using 'var' instead of 'const'

**Low Issues (3)**:
5. `payment.js:9` - console.log in production
6. `payment.js:27` - Comparison with true
7. `tests/test_user_service.py:14` - Bare except clause

**Score**: 80/100

### Test Specialist (Priority 2)

**High Issues (2)**:
1. `payment.js` - No test file exists for critical payment module
2. `tests/test_user_service.py:6` - Test has no assertions

**Medium Issues (2)**:
3. `tests/test_user_service.py` - No setup/teardown
4. `tests/test_user_service.py:14` - Empty catch block hides failures

**Estimated Coverage**: 20%
**Score**: 40/100

### Documentation Reviewer (Priority 4)

**Medium Issues (3)**:
1. `user_service.py:7` - Class missing docstring
2. `user_service.py:14` - authenticate() missing docstring
3. `payment.js:1` - No JSDoc for critical functions

**Low Issues (2)**:
4. `README.md` - Inadequate installation/usage instructions
5. `docs/API.md` - Placeholder documentation

**Score**: 60/100

---

## Aggregated Comprehensive Review

### Overall Score Calculation

```
Weighted Score = (Security × 0.30) + (Performance × 0.25) +
                 (Style × 0.15) + (Tests × 0.20) + (Docs × 0.10)

             = (25 × 0.30) + (75 × 0.25) + (80 × 0.15) +
               (40 × 0.20) + (60 × 0.10)

             = 7.5 + 18.75 + 12.0 + 8.0 + 6.0
             = 52.25/100
```

### Cross-Cutting Concerns

Agent collaboration reveals:

1. **Security + Performance**:
   - N+1 queries both slow AND increase SQL injection attack surface
   - Recommendation: Use ORM with parameterized queries (fixes both)

2. **Tests + Security**:
   - No tests for payment module = security-critical code untested
   - Recommendation: Require tests for all authentication/payment code

3. **Style + Documentation**:
   - Poor style makes code harder to document
   - Missing docstrings compound readability issues

### Merge Decision

```yaml
merge_ready: false
blocking_issues: 3 (critical security)
warnings: 13
suggestions: 7

decision: "request_changes"

priority_fixes:
  1. "Critical: Fix SQL injection in user_service.py:17"
  2. "Critical: Remove hardcoded API key"
  3. "Critical: Upgrade to bcrypt/scrypt for password hashing"
  4. "High: Add tests for payment module (0% coverage)"
  5. "High: Fix N+1 query pattern"
  6. "Medium: Add comprehensive documentation"
```

---

## Swarm Coordination Benefits

### Time Comparison

**Sequential Review** (estimated):
- Security: 60s
- Performance: 45s
- Style: 40s
- Tests: 35s
- Documentation: 30s
- **Total**: 210 seconds (3.5 minutes)

**Parallel Swarm Review** (actual):
- All agents: ~60s (longest individual agent)
- **Total**: ~60 seconds
- **Speedup**: 3.5x faster

### Quality Benefits

1. **Cross-Domain Insights**:
   - Single agent might miss that N+1 queries worsen security
   - Swarm identifies interconnected issues

2. **Comprehensive Coverage**:
   - Each specialist focuses on their domain
   - No single agent can be expert in everything

3. **Consistent Standards**:
   - Security rules applied uniformly
   - Style conventions enforced systematically

---

## JSON Output Validation

```json
{
  "pr_number": 3,
  "pr_title": "Multi-domain test code for swarm review",
  "overall_score": 52.25,
  "merge_ready": false,
  "blocking_issues": 3,
  "warnings": 13,
  "suggestions": 7,
  "agent_reviews": [
    {
      "agent_name": "Security Reviewer",
      "agent_type": "security",
      "score": 25,
      "findings": [ /* 5 findings */ ],
      "execution_time": 1.2,
      "metadata": {
        "critical_issues": 3,
        "high_issues": 2
      }
    },
    {
      "agent_name": "Performance Analyst",
      "agent_type": "performance",
      "score": 75,
      "findings": [ /* 4 findings */ ],
      "execution_time": 0.9,
      "metadata": {
        "bottlenecks_found": 4
      }
    }
    /* ... other agents ... */
  ],
  "timestamp": "2025-11-02T...",
  "execution_time": 2.1
}
```

---

## Validation Criteria

### ✅ Pass Conditions

1. **Swarm Coordination**:
   - All 5 agents spawn successfully
   - Execution is truly parallel (not sequential)
   - Total time < 5 seconds (3x+ speedup)

2. **Issue Detection**:
   - All 27 issues detected (3 critical + 3 high + 13 medium + 8 low)
   - No agent conflicts or duplicate findings
   - Cross-cutting concerns identified

3. **Aggregation Quality**:
   - Overall score accurate (52 ± 3 points)
   - Agent scores properly weighted
   - Findings deduplicated correctly

4. **Comprehensive Report**:
   - JSON output valid
   - All 5 agent reports included
   - Fix suggestions generated
   - Merge decision correct

5. **Memory/Resource**:
   - No memory leaks
   - All agents terminate cleanly
   - No zombie processes

### ❌ Fail Conditions

- Any agent fails to spawn or crashes
- Sequential execution detected (time > 10s)
- Critical issues missed
- Duplicate findings (same issue reported by multiple agents)
- Overall score calculation error > 5 points
- Invalid JSON output

---

## Performance Metrics

**Target Benchmarks**:
- Swarm initialization: < 2s
- Parallel execution: < 5s
- Report aggregation: < 1s
- **Total end-to-end**: < 8s

**Memory Usage**:
- Peak RAM: < 500MB
- No memory leaks after completion

**Agent Efficiency**:
- Each agent processes ~1-2 files/second
- Findings generated in real-time
- No blocking operations

---

## Cleanup

```bash
rm -rf /tmp/test-repo-swarm
```

---

## Test Results

**Date**: _____________
**Executor**: _____________
**Status**: [ ] PASS | [ ] FAIL | [ ] SKIP

**Performance**:
- Swarm Init Time: ____ seconds
- Parallel Execution Time: ____ seconds
- Total Time: ____ seconds
- Speedup Factor: ____x

**Detection Accuracy**:
- Total Issues Detected: ____ / 27
- Critical Detected: ____ / 3
- High Detected: ____ / 3
- Medium Detected: ____ / 13
- Low Detected: ____ / 8

**Swarm Coordination**:
- Agents Spawned: ____ / 5
- Parallel Execution: [ ] Yes [ ] No
- Cross-Cutting Concerns Identified: ____

**Notes**:
-
-
-

**Issues Found**:
-
-
