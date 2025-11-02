# Effective Code Review Best Practices

Comprehensive guide to conducting effective code reviews, both as an author and a reviewer.

## For Code Authors

### 1. Prepare Your PR

#### Before Opening PR

**Self-Review Checklist**:
```markdown
- [ ] Code compiles without warnings
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Code follows project style guide
- [ ] No hardcoded credentials or secrets
- [ ] No commented-out code (use git history instead)
- [ ] Documentation updated (README, API docs)
- [ ] Changelog updated with changes
- [ ] Screenshots added for UI changes
- [ ] Breaking changes clearly documented
```

**Size Matters**:
- **Optimal**: 200-400 lines changed
- **Maximum**: 400-800 lines (split if possible)
- **Too Large**: >800 lines (reviewers lose focus)

**Break Large PRs into Smaller Ones**:
```bash
# Instead of one 2000-line PR
PR #1: Database schema changes (200 lines)
PR #2: Backend API endpoints (300 lines)
PR #3: Frontend components (400 lines)
PR #4: Integration tests (250 lines)
```

#### Write Descriptive PR Description

**Template**:
```markdown
## What
Brief description of changes (1-2 sentences)

## Why
Problem being solved or feature being added

## How
Technical approach taken

## Testing
- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing steps:
  1. Step 1
  2. Step 2
  3. Expected result

## Screenshots / Demos
[If UI changes]

## Breaking Changes
[If any breaking changes, describe migration path]

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No secrets committed
```

**Example Good PR Description**:
```markdown
## What
Add password reset functionality via email

## Why
Users currently cannot reset forgotten passwords, requiring admin intervention (20+ support tickets/month)

## How
1. New POST /auth/reset-password endpoint generates JWT token
2. Token sent via email with 1-hour expiration
3. New PUT /auth/reset-password/:token endpoint validates token and updates password
4. Passwords hashed with bcrypt (cost factor 10)

## Testing
- [x] Unit tests for token generation/validation (12 tests)
- [x] Integration tests for email sending
- [x] Manual testing:
  1. Go to /forgot-password
  2. Enter email
  3. Check email for reset link
  4. Click link, enter new password
  5. Login with new password ✓

## Security Considerations
- Tokens expire after 1 hour
- Tokens invalidated after password change
- Rate limiting: 3 requests per 15 minutes per IP
- Email validation with regex + DNS check

## Breaking Changes
None
```

### 2. Respond to Review Feedback

#### How to Handle Feedback

**DO**:
- ✅ Assume positive intent
- ✅ Ask clarifying questions if unclear
- ✅ Explain reasoning for controversial decisions
- ✅ Mark conversations as resolved after addressing
- ✅ Thank reviewers for catching bugs
- ✅ Be open to alternative approaches

**DON'T**:
- ❌ Get defensive or argumentative
- ❌ Ignore feedback without explanation
- ❌ Make unrelated changes in same PR
- ❌ Force push over reviewer's comments
- ❌ Take criticism personally

**Response Templates**:

```markdown
# Accepting feedback
"Good catch! Fixed in abc1234. Thanks!"

# Disagreeing respectfully
"I considered that approach, but went with X because of Y. What do you think about Z compromise?"

# Asking for clarification
"Could you elaborate on this concern? Are you worried about performance or readability?"

# Explaining decision
"I used approach X instead of Y because our benchmark showed 40% improvement. See benchmark results: [link]"
```

### 3. Keep PR Updated

**Rebase vs Merge**:
```bash
# Rebase (cleaner history, preferred)
git fetch origin
git rebase origin/main
git push --force-with-lease

# Merge (preserves history, safer)
git fetch origin
git merge origin/main
git push
```

**Commit Organization**:
```bash
# ❌ Bad commit history
fix stuff
more fixes
oops
actually works now

# ✅ Good commit history
feat: Add password reset endpoint
test: Add integration tests for password reset
docs: Update API documentation for password reset
refactor: Extract email sending to service
```

---

## For Code Reviewers

### 1. Review Mindset

#### Goals of Code Review

**Primary Goals**:
1. **Catch Bugs**: Find logic errors, edge cases, race conditions
2. **Ensure Security**: Identify vulnerabilities, unsafe patterns
3. **Improve Design**: Suggest better architecture, maintainability
4. **Knowledge Sharing**: Learn from each other, spread best practices
5. **Enforce Standards**: Consistency with team style guide

**NOT Goals**:
- ❌ Show how smart you are
- ❌ Nitpick insignificant details
- ❌ Rewrite code in your personal style
- ❌ Block progress on subjective opinions

#### Review Philosophy

**The Humble Reviewer**:
- Assume the author is smarter than you
- Ask questions instead of making demands
- Understand context before commenting
- Offer suggestions, not mandates
- Praise good code (not just criticize bad code)

**Phrasing Matters**:
```markdown
# ❌ Bad (demanding, aggressive)
"This is wrong. Use X instead."
"Why didn't you just do Y?"
"This code is terrible."

# ✅ Good (questioning, collaborative)
"Could we use X here? It might be more efficient because..."
"What was the reasoning for Y? I'm curious about the trade-offs."
"Have you considered Z approach? It could simplify this logic."
```

### 2. What to Review

#### Security First
```markdown
Priority: CRITICAL
⏱ Time: First 5 minutes

Check for:
- [ ] SQL injection, XSS, command injection
- [ ] Hardcoded secrets or credentials
- [ ] Authentication/authorization bypasses
- [ ] Insecure data transmission (HTTP vs HTTPS)
- [ ] Unsafe deserialization
- [ ] Missing input validation
- [ ] Sensitive data in logs
```

#### Functional Correctness
```markdown
Priority: HIGH
⏱ Time: 10-15 minutes

Check for:
- [ ] Logic errors, off-by-one errors
- [ ] Edge cases (null, empty, max values)
- [ ] Race conditions, thread safety
- [ ] Error handling (try/catch coverage)
- [ ] Correct algorithm implementation
- [ ] Proper state management
```

#### Tests
```markdown
Priority: HIGH
⏱ Time: 5-10 minutes

Check for:
- [ ] Tests actually run and pass
- [ ] New functionality has new tests
- [ ] Tests cover edge cases
- [ ] Tests are independent (no shared state)
- [ ] Tests are readable (AAA pattern)
- [ ] No flaky tests (time-dependent, random)
```

#### Code Quality
```markdown
Priority: MEDIUM
⏱ Time: 10-15 minutes

Check for:
- [ ] Functions <50 lines
- [ ] Classes follow Single Responsibility
- [ ] No code duplication
- [ ] Meaningful variable names
- [ ] Complex logic has comments explaining WHY
- [ ] Consistent with team style guide
```

#### Documentation
```markdown
Priority: LOW
⏱ Time: 5 minutes

Check for:
- [ ] Public APIs have JSDoc/docstrings
- [ ] README updated if needed
- [ ] Changelog updated
- [ ] Complex algorithms explained
- [ ] Migration guide for breaking changes
```

### 3. How to Review

#### Review Process

**1. High-Level Review (10 minutes)**:
- Read PR description
- Check changed files list
- Look for architectural issues
- Identify scope creep

**2. Deep Review (20-30 minutes)**:
- Read code line by line
- Check tests
- Run code locally if complex
- Verify documentation

**3. Final Pass (5 minutes)**:
- Review your own comments (are they helpful?)
- Check for nitpicks (remove if not important)
- Ensure you've asked questions, not just criticized

#### Comment Categories

**CRITICAL (🔴 Blocker)**:
Must be fixed before merge.
```markdown
🔴 CRITICAL: SQL injection vulnerability
This query concatenates user input directly. Use parameterized queries instead.
```

**HIGH (🟠 Strongly Recommended)**:
Should be fixed, but merge possible if good reason.
```markdown
🟠 HIGH: N+1 query problem
This will cause 100 database queries for 100 users. Consider eager loading.
```

**MEDIUM (🟡 Suggestion)**:
Nice to have, but optional.
```markdown
🟡 MEDIUM: Consider extracting to function
This 40-line block could be its own function for readability.
```

**LOW (🟢 Nitpick)**:
Stylistic preference, optional.
```markdown
🟢 NITPICK: Could use const instead of let here
```

**PRAISE (🎉 Compliment)**:
Positive feedback!
```markdown
🎉 PRAISE: Excellent test coverage!
Love the edge cases you thought of here.
```

#### Offer Solutions, Not Just Problems

**❌ Bad Review**:
```markdown
This code is too slow and will cause problems at scale.
```

**✅ Good Review**:
```markdown
This O(n²) loop might become a bottleneck with large datasets.
A few options to consider:

1. Use a Map for O(1) lookups (best for most cases)
2. Pre-sort and use binary search (if data is mostly sorted)
3. Move to database query with index (if data comes from DB)

What do you think about option 1? Happy to pair on implementation if helpful.
```

### 4. Review Timing

**Response Times**:
- **Urgent/Security Fix**: Within 1 hour
- **Regular PR**: Within 4 hours (same day)
- **Large PR**: Within 1 business day
- **RFC/Design Doc**: Within 2 business days

**Review Duration**:
- **Small PR (<100 lines)**: 10-15 minutes
- **Medium PR (100-400 lines)**: 20-30 minutes
- **Large PR (400-800 lines)**: 45-60 minutes
- **Too Large (>800 lines)**: Ask author to split

**When to Pause Review**:
- If you don't understand the domain, ask for context first
- If PR is too large, ask to split
- If you're frustrated, take a break and review later
- If you find critical security issue, stop and escalate immediately

---

## Automated Review Tools

### Static Analysis

**ESLint (JavaScript/TypeScript)**:
```json
{
  "extends": ["airbnb", "plugin:security/recommended"],
  "rules": {
    "complexity": ["error", 10],
    "max-lines-per-function": ["error", 50],
    "max-depth": ["error", 3],
    "no-console": ["warn", { "allow": ["warn", "error"] }]
  }
}
```

**SonarQube**:
- Code smells detection
- Security vulnerability scanning
- Test coverage tracking
- Technical debt quantification

**CodeClimate**:
- Maintainability scoring
- Code duplication detection
- Complexity tracking

### Security Scanning

**Snyk**:
```bash
# Check for vulnerable dependencies
snyk test

# Fix vulnerable dependencies
snyk fix
```

**GitGuardian**:
- Scans for hardcoded secrets
- Monitors for credential leaks
- Integrates with GitHub/GitLab

**OWASP Dependency-Check**:
```bash
# Scan for known CVEs in dependencies
dependency-check --project MyApp --scan ./package.json
```

### Performance Testing

**Lighthouse CI**:
```yaml
# .github/workflows/lighthouse.yml
- name: Run Lighthouse CI
  uses: treosh/lighthouse-ci-action@v9
  with:
    urls: |
      https://staging.example.com
    uploadArtifacts: true
```

**k6 Load Testing**:
```javascript
// load-test.js
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  vus: 100, // 100 virtual users
  duration: '30s',
};

export default function() {
  let res = http.get('https://api.example.com/users');
  check(res, { 'status is 200': (r) => r.status === 200 });
}
```

---

## Review Metrics

### Individual Reviewer Metrics

**Good Metrics**:
- Average review turnaround time
- Percentage of reviews completed within 24 hours
- Bug catch rate (bugs found / total bugs)
- Knowledge sharing score (teaching moments)

**Bad Metrics** (avoid optimizing for these):
- Number of comments per review (encourages nitpicking)
- Number of PRs reviewed (encourages rubber stamping)
- Lines of code reviewed (encourages speed over quality)

### Team Metrics

**Track**:
- Average time from PR open to merge
- Percentage of PRs requiring >2 rounds of review
- Percentage of bugs caught in review vs production
- Test coverage trend over time

**Goals**:
- <24 hours PR turnaround time
- <20% requiring >2 review rounds
- >80% bugs caught in review
- >80% test coverage

---

## Common Review Pitfalls

### For Authors

**1. Defensive About Feedback**
```markdown
❌ Reviewer: "Could we use a Map here for O(1) lookups?"
   Author: "My code works fine. This is just your opinion."

✅ Reviewer: "Could we use a Map here for O(1) lookups?"
   Author: "Good idea! I was focused on correctness, not performance.
           Let me profile first to see if it's a bottleneck."
```

**2. Scope Creep**
```markdown
❌ PR: "Add login endpoint" + refactor entire auth system + update UI theme

✅ PR: "Add login endpoint" (one focused change)
   Next PR: "Refactor auth system"
   Next PR: "Update UI theme"
```

**3. Force Pushing Over Comments**
```bash
# ❌ Bad: Destroys reviewer's context
git push --force

# ✅ Good: Preserves review history
git push --force-with-lease  # Only if you're sure
# Or better: just merge main instead of rebasing
```

### For Reviewers

**1. Nitpicking Style**
```markdown
❌ "Use single quotes instead of double quotes"
   (If project has no style guide)

✅ Setup Prettier/ESLint to enforce automatically
```

**2. Blocking on Subjective Opinions**
```markdown
❌ "I don't like this approach. Use my way or I won't approve."

✅ "I would have used approach X because of Y, but your approach Z
    also works. Both are valid trade-offs."
```

**3. Reviewing Too Fast**
```markdown
❌ 500-line PR reviewed in 2 minutes
   "LGTM 👍" (obviously didn't read)

✅ Take proper time, ask questions, actually understand code
```

**4. Rewriting in Your Style**
```markdown
❌ "Change this entire function to how I would write it"
   (Same behavior, just personal preference)

✅ Focus on correctness, security, performance, not personal style
```

---

## Review Checklist

### Mandatory Checks
```markdown
- [ ] No security vulnerabilities (SQL injection, XSS, etc.)
- [ ] No hardcoded secrets or credentials
- [ ] All tests pass
- [ ] New functionality has tests
- [ ] No breaking changes without migration guide
- [ ] Error handling covers failure cases
- [ ] Code follows team style guide
```

### Recommended Checks
```markdown
- [ ] Functions <50 lines
- [ ] Cyclomatic complexity <10
- [ ] No code duplication
- [ ] Meaningful variable names
- [ ] Complex logic has explanatory comments
- [ ] Public APIs have documentation
- [ ] README/docs updated if needed
```

### Optional Checks
```markdown
- [ ] Performance optimizations considered
- [ ] Accessibility considered (if UI)
- [ ] Mobile-responsive (if web UI)
- [ ] Internationalization considered
- [ ] Logging added for debugging
```

---

## Example Review Comments

### Security
```markdown
🔴 CRITICAL: SQL Injection Vulnerability
Line 45: Concatenating user input in SQL query allows SQL injection.

Vulnerable code:
```js
db.query(`SELECT * FROM users WHERE id=${userId}`)
```

Fix:
```js
db.query('SELECT * FROM users WHERE id=?', [userId])
```

Resources:
- OWASP SQL Injection: https://owasp.org/...
- Parameterized queries guide: https://...
```

### Performance
```markdown
🟠 HIGH: N+1 Query Problem
Lines 78-85: Loading posts for each user in a loop causes N+1 queries.

Current: 101 queries for 100 users
With eager loading: 2 queries total

Suggestion:
```js
// Instead of:
for (let user of users) {
  user.posts = await db.posts.find({ userId: user.id });
}

// Use eager loading:
const users = await db.users.find({
  include: [{ model: db.posts }]
});
```

Benchmark: https://...
```

### Code Quality
```markdown
🟡 MEDIUM: Consider Extracting to Function
Lines 120-160: This 40-line block could be its own function.

Benefits:
- Easier to test in isolation
- Reusable if needed elsewhere
- Makes parent function more readable

Suggestion:
```js
function validateAndProcessPayment(order) {
  // Extract this 40-line logic here
}

// Then in main function:
const result = validateAndProcessPayment(order);
```

Not blocking, but would improve maintainability. What do you think?
```

### Praise
```markdown
🎉 PRAISE: Excellent Error Handling
Love the comprehensive error handling in lines 200-220!

The specific error messages will make debugging much easier:
- Distinguishes network errors from validation errors
- Includes context (user ID, timestamp)
- Doesn't expose sensitive data

This is exactly the level of detail we should aim for. Great work!
```

---

## Resources

### Books
- "Code Complete" by Steve McConnell
- "The Art of Readable Code" by Boswell & Foucher
- "Clean Code" by Robert C. Martin
- "Refactoring" by Martin Fowler

### Online Resources
- [Google Code Review Guidelines](https://google.github.io/eng-practices/review/)
- [Microsoft Code Review Best Practices](https://docs.microsoft.com/en-us/azure/devops/repos/git/pull-requests)
- [GitHub Code Review Best Practices](https://github.blog/2015-01-21-how-to-write-the-perfect-pull-request/)
- [OWASP Code Review Guide](https://owasp.org/www-project-code-review-guide/)

### Tools
- **ESLint**: JavaScript linting
- **SonarQube**: Code quality platform
- **Snyk**: Dependency vulnerability scanning
- **GitGuardian**: Secret detection
- **Codacy**: Automated code review
- **DeepSource**: Static analysis
