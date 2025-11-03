# Example 1: Complete PR Review Workflow

## Scenario Overview

**Context**: A developer has submitted PR #456 adding user authentication to a Node.js/Express API. The PR includes JWT token implementation, password hashing, and login/signup endpoints.

**Team Requirements**:
- Security best practices (OWASP compliance)
- Test coverage ≥85%
- Performance benchmarks for auth endpoints
- Documentation for API endpoints
- Code style consistency (ESLint/Prettier)

**PR Details**:
- Repository: `company/api-backend`
- Branch: `feature/user-authentication`
- Files Changed: 8 files (+847, -23 lines)
- Author: `@developer-jane`

---

## Step 1: Initialize Review Swarm

### Command Execution

```bash
# Navigate to repository
cd ~/projects/api-backend

# Initialize hierarchical swarm for comprehensive review
npx claude-flow@alpha swarm init \
  --topology hierarchical \
  --max-agents 6 \
  --strategy specialized

# Expected Output:
# ✓ Swarm initialized: swarm-auth-review-20251102
# ✓ Topology: Hierarchical (coordinator → 5 specialists)
# ✓ Agents: 6/6 ready
# ✓ Session ID: auth-pr-456-review
```

### Agent Spawning Strategy

```javascript
// Coordinator spawns 5 specialist reviewers in parallel
[Single Message - Parallel Agent Execution]:

  // Security specialist (HIGHEST PRIORITY for auth code)
  Task("Security Auditor",
    "Review authentication implementation for OWASP Top 10 vulnerabilities. Check JWT secret management, password hashing (bcrypt rounds), SQL injection risks, and rate limiting on auth endpoints.",
    "analyst")

  // Code quality specialist
  Task("Code Quality Reviewer",
    "Review code structure, ESLint compliance, error handling patterns, async/await usage, and separation of concerns. Verify middleware architecture and controller/service separation.",
    "analyst")

  // Testing specialist
  Task("Test Engineer",
    "Analyze test coverage for auth flows. Verify unit tests for services, integration tests for endpoints, edge cases (invalid tokens, expired sessions), and mock strategies.",
    "tester")

  // Performance specialist
  Task("Performance Analyst",
    "Benchmark authentication endpoints. Check database query optimization, bcrypt rounds impact on response time, token generation performance, and memory usage.",
    "optimizer")

  // Documentation specialist
  Task("Documentation Reviewer",
    "Review API documentation completeness. Verify OpenAPI/Swagger specs, endpoint descriptions, error response examples, and authentication flow diagrams.",
    "researcher")
```

---

## Step 2: Automated Analysis Execution

### Security Review Output

**Agent**: Security Auditor
**Duration**: 2m 34s
**Findings**: 3 issues (1 CRITICAL, 2 HIGH)

```markdown
## Security Audit Report - PR #456

### CRITICAL Issues (1)

❌ **JWT Secret Hardcoded in Source**
- **File**: `src/auth/jwt.service.js:12`
- **Issue**: JWT secret stored as string literal
- **Code**:
  ```javascript
  const secret = 'my-super-secret-key-123'; // CRITICAL
  const token = jwt.sign(payload, secret, { expiresIn: '1h' });
  ```
- **Risk**: Secret exposed in version control, attackers can forge tokens
- **Fix**: Use environment variables
  ```javascript
  const secret = process.env.JWT_SECRET;
  if (!secret) throw new Error('JWT_SECRET not configured');
  const token = jwt.sign(payload, secret, { expiresIn: '1h' });
  ```
- **References**: OWASP A02:2021 – Cryptographic Failures

### HIGH Issues (2)

⚠️ **Bcrypt Rounds Too Low**
- **File**: `src/auth/password.service.js:8`
- **Issue**: Only 8 bcrypt rounds (industry standard: 12-14)
- **Code**:
  ```javascript
  const hash = await bcrypt.hash(password, 8); // TOO LOW
  ```
- **Risk**: Vulnerable to brute-force attacks with modern GPUs
- **Fix**: Increase to 12 rounds
  ```javascript
  const BCRYPT_ROUNDS = parseInt(process.env.BCRYPT_ROUNDS || '12', 10);
  const hash = await bcrypt.hash(password, BCRYPT_ROUNDS);
  ```

⚠️ **No Rate Limiting on Login Endpoint**
- **File**: `src/routes/auth.routes.js:15`
- **Issue**: Missing rate limiting middleware
- **Risk**: Susceptible to credential stuffing and brute-force attacks
- **Fix**: Add express-rate-limit
  ```javascript
  const rateLimit = require('express-rate-limit');

  const loginLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // 5 attempts per window
    message: 'Too many login attempts, please try again later'
  });

  router.post('/login', loginLimiter, authController.login);
  ```

### PASSED Checks ✓

- [x] Password validation (min 8 chars, complexity requirements)
- [x] SQL parameterization (no injection risks)
- [x] HTTPS enforcement in production
- [x] Token expiration configured (1 hour)
- [x] Sensitive data not logged
```

---

### Performance Review Output

**Agent**: Performance Analyst
**Duration**: 3m 12s
**Findings**: 2 optimization opportunities

```markdown
## Performance Benchmark Report - PR #456

### Endpoint Response Times

| Endpoint | Method | Avg (ms) | P95 (ms) | P99 (ms) | Status |
|----------|--------|----------|----------|----------|--------|
| `/auth/signup` | POST | 287 | 412 | 523 | ⚠️ SLOW |
| `/auth/login` | POST | 245 | 356 | 445 | ⚠️ SLOW |
| `/auth/verify` | GET | 34 | 48 | 62 | ✓ GOOD |

### Bottleneck Analysis

⚠️ **Bcrypt Performance Impact**
- **Issue**: Synchronous database call + bcrypt = 250ms+ response time
- **Current Code** (`src/auth/auth.service.js:23-26`):
  ```javascript
  async signup(email, password) {
    const hashedPassword = await bcrypt.hash(password, 12); // ~200ms
    const user = await db.users.create({ email, hashedPassword }); // ~50ms
    return user;
  }
  ```
- **Optimization**: Use worker threads for bcrypt
  ```javascript
  const { Worker } = require('worker_threads');

  async signup(email, password) {
    // Offload bcrypt to worker thread (non-blocking)
    const hashedPassword = await this.hashInWorker(password);
    const user = await db.users.create({ email, hashedPassword });
    return user;
  }

  hashInWorker(password) {
    return new Promise((resolve, reject) => {
      const worker = new Worker('./workers/bcrypt.worker.js');
      worker.postMessage({ password, rounds: 12 });
      worker.on('message', resolve);
      worker.on('error', reject);
    });
  }
  ```
- **Expected Improvement**: 60-70% reduction in P95 latency

### Database Query Optimization

⚠️ **Missing Index on Email Column**
- **Issue**: Login queries scan full users table (10K+ rows)
- **Query**: `SELECT * FROM users WHERE email = ?`
- **Execution Plan**: Seq Scan on users (cost=0.00..180.00 rows=10000)
- **Fix**: Add unique index
  ```sql
  CREATE UNIQUE INDEX idx_users_email ON users(email);
  ```
- **Expected Improvement**: 95% faster lookups (180ms → 8ms)

### Recommendations

1. Implement worker thread bcrypt hashing → **60% latency reduction**
2. Add database index on email column → **95% faster queries**
3. Consider caching JWT public keys (if using RSA) → **5% faster verification**
4. Use connection pooling (max 20 connections) → **10% throughput increase**

**Target Metrics After Optimizations**:
- `/auth/signup`: 100ms avg, 150ms P95
- `/auth/login`: 80ms avg, 120ms P95
```

---

## Step 3: Consolidated Review Report

### GitHub PR Comment (Auto-Generated)

```markdown
## 🤖 AI Swarm Code Review - PR #456

**Review Duration**: 8m 47s
**Reviewers**: 5 specialist agents (Security, Quality, Testing, Performance, Docs)
**Overall Status**: ⚠️ **CHANGES REQUESTED**

---

### 🔴 CRITICAL Issues (Must Fix Before Merge)

1. **JWT Secret Hardcoded** - `src/auth/jwt.service.js:12`
   - **Risk**: Secret exposed in Git history → token forgery attacks
   - **Fix**: Use `process.env.JWT_SECRET` with validation
   - **Reference**: OWASP A02:2021

---

### ⚠️ HIGH Priority Issues (Recommended Fixes)

2. **Bcrypt Rounds Too Low** - `src/auth/password.service.js:8`
   - Use 12 rounds instead of 8 (industry standard)

3. **No Rate Limiting** - `src/routes/auth.routes.js:15`
   - Add express-rate-limit (5 attempts / 15 min window)

4. **Missing Database Index** - Performance impact on login
   - Add `CREATE UNIQUE INDEX idx_users_email ON users(email)`

5. **Test Coverage Below Target** - Current: 78%, Target: 85%
   - Missing tests: token expiration, invalid credentials, duplicate email

---

### ✅ PASSED Checks (32/37)

- [x] Password validation and complexity
- [x] SQL injection protection
- [x] HTTPS enforcement
- [x] Error handling patterns
- [x] ESLint compliance (0 errors)
- [x] API documentation complete
- [x] Controller/service separation

---

### 📊 Metrics

| Category | Score | Target | Status |
|----------|-------|--------|--------|
| Security | 7/10 | 9/10 | ⚠️ |
| Code Quality | 9/10 | 8/10 | ✅ |
| Test Coverage | 78% | 85% | ⚠️ |
| Performance | 6/10 | 8/10 | ⚠️ |
| Documentation | 10/10 | 8/10 | ✅ |

---

### 🔧 Suggested Changes

<details>
<summary>View Auto-Fix Suggestions (3 files)</summary>

**File: `src/auth/jwt.service.js`**
```diff
- const secret = 'my-super-secret-key-123';
+ const secret = process.env.JWT_SECRET;
+ if (!secret) throw new Error('JWT_SECRET environment variable not set');
```

**File: `src/auth/password.service.js`**
```diff
- const hash = await bcrypt.hash(password, 8);
+ const BCRYPT_ROUNDS = parseInt(process.env.BCRYPT_ROUNDS || '12', 10);
+ const hash = await bcrypt.hash(password, BCRYPT_ROUNDS);
```

**File: `src/routes/auth.routes.js`**
```diff
+ const rateLimit = require('express-rate-limit');
+ const loginLimiter = rateLimit({
+   windowMs: 15 * 60 * 1000,
+   max: 5,
+   message: 'Too many login attempts, please try again later'
+ });
+
- router.post('/login', authController.login);
+ router.post('/login', loginLimiter, authController.login);
```

</details>

---

### 📝 Next Steps

1. **Developer**: Address CRITICAL issue #1 (JWT secret)
2. **Developer**: Fix HIGH priority issues #2-5
3. **CI/CD**: Re-run tests after fixes (target: 85%+ coverage)
4. **Reviewer**: Final approval after changes applied

**Estimated Time to Fix**: 2-3 hours

---

*Generated by AI Swarm Code Review | Powered by Claude Flow*
```

---

## Step 4: Developer Applies Fixes

### Updated Files (Developer Commits)

```bash
# Developer makes changes based on review
git checkout feature/user-authentication

# Apply auto-fix suggestions
# ... edit files ...

git add src/auth/jwt.service.js src/auth/password.service.js src/routes/auth.routes.js
git commit -m "fix: address security and performance issues from AI review

- Move JWT secret to environment variable
- Increase bcrypt rounds to 12
- Add rate limiting to login endpoint
- Add database index migration for users.email
- Add missing test cases for edge scenarios

Resolves feedback from PR #456 review"

git push origin feature/user-authentication
```

---

## Step 5: Re-Review and Approval

### Automated Re-Scan

```bash
# AI swarm automatically re-scans after new commits
# Triggered by GitHub webhook or manual command:

npx claude-flow@alpha github review-pr \
  --repo company/api-backend \
  --pr 456 \
  --mode incremental

# Output:
# ✓ Detected new commits (2 files changed)
# ✓ Re-scanning modified files only
# ✓ Security: PASSED (10/10)
# ✓ Performance: PASSED (9/10)
# ✓ Tests: PASSED (87% coverage)
# ✓ All CRITICAL issues resolved
# ✓ Ready for merge ✅
```

### Final Approval Comment

```markdown
## ✅ AI Swarm Re-Review - APPROVED

**Status**: All issues resolved
**New Commits**: 2 (SHA: `a3f2b1c`, `d4e5f6g`)

### Fixed Issues

- [x] #1: JWT secret now in environment variable
- [x] #2: Bcrypt rounds increased to 12
- [x] #3: Rate limiting added (5/15min)
- [x] #4: Database index migration created
- [x] #5: Test coverage now 87% (target: 85%)

### Final Metrics

| Category | Score | Change | Status |
|----------|-------|--------|--------|
| Security | 10/10 | +3 | ✅ |
| Performance | 9/10 | +3 | ✅ |
| Test Coverage | 87% | +9% | ✅ |

**Recommendation**: ✅ **APPROVE AND MERGE**

---

*Reviewed by: Security Auditor, Performance Analyst, Test Engineer*
```

---

## Outcomes and Benefits

### Time Savings

- **Manual Review Time**: ~45 minutes (senior engineer)
- **AI Swarm Review Time**: 8m 47s (initial) + 3m 12s (re-review) = **11m 59s**
- **Time Saved**: 73% reduction
- **Developer Time to Fix**: 2h 15m (with clear guidance)

### Quality Improvements

- **Security Score**: 7/10 → 10/10 (+43%)
- **Test Coverage**: 78% → 87% (+9 percentage points)
- **Performance**: Login latency 245ms → 95ms (61% improvement)
- **Issues Caught**: 5 issues (1 CRITICAL) before production

### Team Impact

- **Developer Feedback**: Clear, actionable suggestions with code examples
- **Learning Opportunity**: Security best practices (OWASP) + performance optimization
- **Consistency**: Same review standards across all PRs
- **Scalability**: Can review 10+ PRs simultaneously

---

## Tips for Effective PR Reviews

### 1. Customize Agent Expertise

```bash
# For frontend PRs, spawn different specialists
Task("Accessibility Specialist", "WCAG 2.1 AA compliance check", "analyst")
Task("Performance Analyst", "Lighthouse score + bundle size", "optimizer")
Task("Component Reviewer", "React best practices + hooks usage", "analyst")
```

### 2. Integrate with CI/CD

```yaml
# .github/workflows/ai-review.yml
name: AI Swarm Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  ai-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run AI Swarm Review
        run: |
          npx claude-flow@alpha github review-pr \
            --repo ${{ github.repository }} \
            --pr ${{ github.event.pull_request.number }} \
            --auto-comment
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

### 3. Use Memory for Context

```bash
# Store project-specific review guidelines
npx claude-flow@alpha memory store \
  --key "review-guidelines/api-backend" \
  --value "Security: OWASP compliance mandatory. Performance: <200ms P95. Tests: 85%+ coverage."

# Agents auto-retrieve during review
npx claude-flow@alpha memory retrieve --key "review-guidelines/api-backend"
```

### 4. Enable Auto-Fix Mode

```bash
# Generate not just comments, but actual fix commits
npx claude-flow@alpha github review-pr \
  --repo company/api-backend \
  --pr 456 \
  --auto-fix \
  --create-commits

# Creates branch: ai-fixes/pr-456
# Applies safe auto-fixes (linting, formatting, simple refactors)
# Developer reviews and merges fix branch
```

---

## Conclusion

This example demonstrates a complete GitHub code review workflow using AI swarm coordination. The key benefits are:

1. **Speed**: 73% faster than manual reviews
2. **Quality**: Catches security and performance issues consistently
3. **Learning**: Provides educational feedback with references
4. **Scalability**: Reviews multiple PRs in parallel
5. **Integration**: Works seamlessly with existing CI/CD pipelines

By combining specialist agents (security, performance, testing), you achieve comprehensive reviews that match or exceed human expert quality, while freeing senior engineers to focus on architecture and mentorship.
