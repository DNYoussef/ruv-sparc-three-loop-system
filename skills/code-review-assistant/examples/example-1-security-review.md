# Example 1: Security-Focused Code Review

This example demonstrates the Code Review Assistant detecting and fixing a critical SQL injection vulnerability in an authentication API.

## Scenario

**PR #456**: Add user login endpoint
**Files Changed**: `src/api/auth.js`
**Lines Added**: 45
**Focus Area**: Security

## Original Code (Vulnerable)

```javascript
// src/api/auth.js
const express = require('express');
const mysql = require('mysql');
const router = express.Router();

const db = mysql.createConnection({
  host: 'localhost',
  user: 'root',
  password: process.env.DB_PASSWORD, // Hardcoded in .env
  database: 'users'
});

router.post('/login', (req, res) => {
  const { username, password } = req.body;

  // VULNERABILITY: SQL Injection
  const query = `SELECT * FROM users WHERE username='${username}' AND password='${password}'`;

  db.query(query, (err, results) => {
    if (err) {
      return res.status(500).json({ error: 'Database error' });
    }

    if (results.length > 0) {
      // VULNERABILITY: Plaintext password comparison
      res.json({ success: true, user: results[0] });
    } else {
      res.status(401).json({ error: 'Invalid credentials' });
    }
  });
});

module.exports = router;
```

## Code Review Assistant Execution

```bash
$ code-review-assistant 456 security

================================================================
Code Review Assistant: PR #456
================================================================

[1/8] Gathering PR information...
PR: Add user login endpoint
Files changed: 1

[2/8] Initializing multi-agent review swarm...
✓ Swarm initialized (mesh topology, 5 agents)
✓ Security Specialist spawned

[3/8] Executing specialized reviews in parallel...
  → Security Specialist reviewing...

[4/8] Running complete quality audit...
✓ Security scan complete: 3 critical issues detected

[5/8] Aggregating review findings...
Overall Score: 32/100

[6/8] Generating fix suggestions with Codex...
✓ 3 fix suggestions generated

[7/8] Assessing merge readiness...
Merge Ready: ⚠️ No
Critical Security Issues: 3

[8/8] Creating review comment...
✓ Review comment posted to PR #456
✓ Changes requested

================================================================
Code Review Complete!
================================================================

Overall Score: 32/100
Merge Ready: false
Decision: request_changes

Review artifacts in: pr-review-456/
```

## Security Review Output

### Critical Issues Detected

#### 1. SQL Injection Vulnerability (CRITICAL)
**File**: `src/api/auth.js`
**Line**: 13
**Severity**: CRITICAL
**CVSS Score**: 9.8 (Critical)

**Issue**:
```javascript
const query = `SELECT * FROM users WHERE username='${username}' AND password='${password}'`;
```

**Risk**: Allows arbitrary SQL execution. Attacker can bypass authentication, dump database, or execute admin commands.

**Attack Example**:
```bash
curl -X POST http://api.example.com/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin\" OR \"1\"=\"1", "password": "anything"}'
# Result: Authentication bypass
```

**Fix Suggestion**:
```javascript
// Use parameterized queries (prepared statements)
const query = 'SELECT * FROM users WHERE username=? AND password=?';
db.query(query, [username, password], (err, results) => {
  // ... rest of code
});
```

**Reasoning**: Parameterized queries prevent SQL injection by treating user input as data, not executable code.

---

#### 2. Plaintext Password Storage (HIGH)
**File**: `src/api/auth.js`
**Line**: 13
**Severity**: HIGH
**OWASP**: A02:2021 - Cryptographic Failures

**Issue**: Password compared in plaintext without hashing.

**Risk**: Database breach exposes all user passwords. No forward secrecy.

**Fix Suggestion**:
```javascript
const bcrypt = require('bcrypt');

// During user registration
const hashedPassword = await bcrypt.hash(password, 10);
// Store hashedPassword in database

// During login
const query = 'SELECT * FROM users WHERE username=?';
db.query(query, [username], async (err, results) => {
  if (results.length > 0) {
    const validPassword = await bcrypt.compare(password, results[0].password);
    if (validPassword) {
      res.json({ success: true, user: results[0] });
    } else {
      res.status(401).json({ error: 'Invalid credentials' });
    }
  }
});
```

**Reasoning**: bcrypt with salt (cost factor 10) provides industry-standard password hashing.

---

#### 3. Sensitive Data Exposure (MEDIUM)
**File**: `src/api/auth.js`
**Line**: 20
**Severity**: MEDIUM
**OWASP**: A01:2021 - Broken Access Control

**Issue**: Returns entire user object including sensitive fields.

**Risk**: Exposes email, phone, internal IDs to client.

**Fix Suggestion**:
```javascript
if (results.length > 0) {
  const { id, username, email } = results[0]; // Whitelist fields
  res.json({
    success: true,
    user: { id, username, email } // Only return needed fields
  });
}
```

**Reasoning**: Principle of least privilege - only return data client needs.

---

## Complete Fixed Code

```javascript
// src/api/auth.js (SECURE VERSION)
const express = require('express');
const mysql = require('mysql2/promise'); // Use promise-based driver
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const router = express.Router();

// Use connection pooling for production
const pool = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

router.post('/login', async (req, res) => {
  try {
    const { username, password } = req.body;

    // Input validation
    if (!username || !password) {
      return res.status(400).json({ error: 'Username and password required' });
    }

    // FIXED: Parameterized query prevents SQL injection
    const [rows] = await pool.execute(
      'SELECT id, username, email, password FROM users WHERE username=?',
      [username]
    );

    if (rows.length === 0) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const user = rows[0];

    // FIXED: bcrypt password verification
    const validPassword = await bcrypt.compare(password, user.password);

    if (!validPassword) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Generate JWT token
    const token = jwt.sign(
      { id: user.id, username: user.username },
      process.env.JWT_SECRET,
      { expiresIn: '24h' }
    );

    // FIXED: Only return non-sensitive fields
    res.json({
      success: true,
      user: {
        id: user.id,
        username: user.username,
        email: user.email
      },
      token
    });

  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;
```

## Review Summary

| Category | Score | Issues |
|----------|-------|--------|
| Security | 32/100 | 3 critical, 0 high, 1 medium |
| Performance | 85/100 | 1 warning (connection pooling) |
| Style | 78/100 | 2 suggestions (error handling) |
| Tests | 0/100 | No tests found |
| Documentation | 45/100 | Missing JSDoc |

**Overall Score**: 32/100 (Failed)
**Merge Decision**: REQUEST_CHANGES
**Blocking Issues**: 3 critical security vulnerabilities

## Required Actions Before Merge

1. Fix SQL injection (parameterized queries)
2. Implement bcrypt password hashing
3. Remove sensitive data from response
4. Add unit tests for auth endpoints
5. Add integration tests with security scenarios
6. Document API with JSDoc/OpenAPI

## Additional Security Recommendations

1. **Rate Limiting**: Prevent brute force attacks
   ```javascript
   const rateLimit = require('express-rate-limit');
   const loginLimiter = rateLimit({
     windowMs: 15 * 60 * 1000, // 15 minutes
     max: 5 // 5 attempts
   });
   router.post('/login', loginLimiter, async (req, res) => { ... });
   ```

2. **HTTPS Only**: Enforce secure transport
3. **Input Sanitization**: Validate all user inputs
4. **Security Headers**: Use Helmet.js middleware
5. **Audit Logging**: Log all authentication attempts
6. **Multi-Factor Authentication**: Add 2FA support

## Testing Recommendations

```javascript
// tests/auth.test.js
describe('POST /login', () => {
  test('should prevent SQL injection', async () => {
    const response = await request(app)
      .post('/login')
      .send({ username: "admin' OR '1'='1", password: 'anything' });
    expect(response.status).toBe(401);
  });

  test('should reject weak passwords', async () => {
    const response = await request(app)
      .post('/register')
      .send({ username: 'user', password: '123' });
    expect(response.status).toBe(400);
  });

  test('should hash passwords', async () => {
    // Verify password is hashed in database
    const user = await User.findOne({ username: 'testuser' });
    expect(user.password).not.toBe('plaintext');
    expect(user.password).toMatch(/^\$2[aby]\$/); // bcrypt format
  });
});
```

## Conclusion

This PR introduces **3 critical security vulnerabilities** that must be fixed before merge. The Code Review Assistant detected all issues, provided specific fix suggestions, and recommended comprehensive security improvements. After applying fixes, re-run review to verify security posture.
