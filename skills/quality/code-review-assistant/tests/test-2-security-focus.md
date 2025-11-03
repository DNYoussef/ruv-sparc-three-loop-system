# Test 2: Security-Focused Review

**Test ID**: TEST-CRA-002
**Skill**: code-review-assistant
**Tier**: Gold
**Focus**: Deep security scanning with Security Reviewer agent

---

## Test Objective

Validate that code-review-assistant can:
1. Execute comprehensive security scan
2. Detect all OWASP Top 10 vulnerabilities
3. Prioritize security findings correctly
4. Block merge on critical security issues
5. Generate actionable remediation steps

---

## Test Setup

### Test Repository with Security Vulnerabilities

```bash
mkdir -p /tmp/test-repo-security
cd /tmp/test-repo-security
git init

# 1. SQL Injection (Critical)
cat > database.py <<'EOF'
def get_user(user_id):
    # SQL injection via string concatenation
    query = "SELECT * FROM users WHERE id=" + user_id
    return execute(query)

def search_users(term):
    # SQL injection in f-string
    query = f"SELECT * FROM users WHERE name LIKE '%{term}%'"
    return db.query(query)
EOF

# 2. XSS Vulnerabilities (Critical)
cat > render.jsx <<'EOF'
import React from 'react';

function UserProfile({ user }) {
  // XSS via dangerouslySetInnerHTML
  return (
    <div dangerouslySetInnerHTML={{ __html: user.bio }} />
  );
}

function Comments({ comments }) {
  // XSS via innerHTML
  comments.forEach(c => {
    document.getElementById('comments').innerHTML += c.text;
  });
}
EOF

# 3. Hardcoded Secrets (Critical)
cat > config.py <<'EOF'
# AWS credentials in code
AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# Database password
DB_PASSWORD = "super_secret_password_123"

# API keys
STRIPE_API_KEY = "sk_live_51H5..."
OPENAI_API_KEY = "sk-proj-..."
EOF

# 4. Weak Cryptography (High)
cat > crypto.py <<'EOF'
import hashlib

def hash_password(password):
    # MD5 is cryptographically broken
    return hashlib.md5(password.encode()).hexdigest()

def encrypt_data(data):
    # SHA1 is also weak
    return hashlib.sha1(data.encode()).hexdigest()
EOF

# 5. Command Injection (Critical)
cat > execute.py <<'EOF'
import subprocess
import os

def run_command(user_input):
    # Command injection via shell=True
    subprocess.run(f"ls {user_input}", shell=True)

def eval_code(code):
    # Arbitrary code execution
    eval(code)
EOF

# 6. Insecure Deserialization (High)
cat > deserialize.py <<'EOF'
import pickle
import yaml

def load_data(data):
    # Insecure pickle deserialization
    return pickle.loads(data)

def load_config(config_str):
    # Unsafe YAML loading
    return yaml.load(config_str)
EOF

# 7. Missing CSRF Protection (Medium)
cat > form.html <<'EOF'
<form method="POST" action="/api/transfer">
  <!-- No CSRF token -->
  <input type="text" name="amount" />
  <input type="text" name="recipient" />
  <button type="submit">Transfer</button>
</form>
EOF

# 8. Insecure Random (High)
cat > random_gen.js <<'EOF'
function generateToken() {
  // Math.random() not cryptographically secure
  return Math.random().toString(36).substring(7);
}

function generateSessionId() {
  return Math.random() * 1000000;
}
EOF

git add .
git commit -m "Security vulnerabilities for testing"
```

---

## Test Execution

### Step 1: Run Security-Only Review

```bash
cd /tmp/test-repo-security

# Focus only on security
code-review-assistant --pr-number 2 \
  --changed-files "database.py,render.jsx,config.py,crypto.py,execute.py,deserialize.py,form.html,random_gen.js" \
  --focus-areas "security" \
  --deep-scan true
```

### Step 2: Run Deep Security Scan Script

```bash
# Execute security scan script directly
bash ../code-review-assistant/resources/scripts/security_scan.sh . security-results.json true
```

---

## Expected Results

### Critical Vulnerabilities (Must Detect)

1. **SQL Injection** (3 instances)
   - `database.py:3` - String concatenation
   - `database.py:7` - f-string interpolation
   - CWE-89

2. **XSS** (2 instances)
   - `render.jsx:6` - dangerouslySetInnerHTML
   - `render.jsx:13` - innerHTML assignment
   - CWE-79

3. **Hardcoded Secrets** (5 instances)
   - `config.py:2` - AWS Access Key
   - `config.py:3` - AWS Secret Key
   - `config.py:6` - Database password
   - `config.py:9` - Stripe API key
   - `config.py:10` - OpenAI API key
   - CWE-798

4. **Command Injection** (2 instances)
   - `execute.py:6` - shell=True vulnerability
   - `execute.py:9` - eval() usage
   - CWE-78, CWE-95

### High-Priority Vulnerabilities

1. **Weak Cryptography** (2 instances)
   - `crypto.py:5` - MD5 usage
   - `crypto.py:9` - SHA1 usage
   - CWE-327

2. **Insecure Deserialization** (2 instances)
   - `deserialize.py:6` - pickle.loads
   - `deserialize.py:10` - yaml.load
   - CWE-502

3. **Insecure Random** (2 instances)
   - `random_gen.js:3` - Math.random() for tokens
   - `random_gen.js:7` - Math.random() for sessions
   - CWE-338

### Medium-Priority Issues

1. **CSRF Missing** (1 instance)
   - `form.html:1` - POST form without token
   - CWE-352

---

## Security Score Calculation

```
Total Critical Issues: 12
Total High Issues: 6
Total Medium Issues: 1

Security Score = 100 - (12 × 30) - (6 × 15) - (1 × 5)
              = 100 - 360 - 90 - 5
              = -355 (capped at 0)
              = 0/100
```

---

## Merge Decision Logic

```yaml
merge_ready: false
blocking_issues: 12
decision: "request_changes"
status: "CRITICAL - DO NOT MERGE"

reasoning:
  - "12 critical security vulnerabilities detected"
  - "SQL injection, XSS, and hardcoded secrets must be fixed"
  - "Command injection and eval() present code execution risks"
  - "Code is not safe for production deployment"

required_actions:
  - "Fix all SQL injection with parameterized queries"
  - "Sanitize all user input before rendering"
  - "Move all secrets to environment variables"
  - "Replace weak crypto with SHA-256 or stronger"
  - "Remove eval() and shell=True"
  - "Use secure random generators for tokens"
```

---

## JSON Output Validation

```json
{
  "agent": "Security Reviewer",
  "score": 0,
  "status": "FAIL",
  "summary": {
    "critical_issues": 12,
    "high_issues": 6,
    "medium_issues": 1,
    "low_issues": 0,
    "total_issues": 19
  },
  "issues": [
    {
      "severity": "critical",
      "category": "sql_injection",
      "message": "Potential SQL injection vulnerability",
      "file": "database.py",
      "line": 3,
      "suggestion": "Use parameterized queries"
    }
  ],
  "merge_recommendation": "BLOCK"
}
```

---

## Validation Criteria

### ✅ Pass Conditions

1. **Detection Accuracy**
   - All 19 vulnerabilities detected (12 critical + 6 high + 1 medium)
   - Zero false negatives for critical issues
   - Correct CWE classification

2. **Severity Assignment**
   - SQL injection marked as critical
   - XSS marked as critical
   - Hardcoded secrets marked as critical
   - Weak crypto marked as high

3. **Remediation Guidance**
   - Each finding has actionable suggestion
   - CWE references provided
   - Tool recommendations included

4. **Merge Decision**
   - `merge_ready = false`
   - Status = "FAIL" or "CRITICAL"
   - Clear blocking issues listed

5. **Performance**
   - Deep scan completes in < 30 seconds
   - JSON output valid

### ❌ Fail Conditions

- Any critical vulnerability missed
- Incorrect severity classification
- False positives > 20%
- Missing remediation guidance
- Merge decision allows critical issues

---

## Additional Tools Validation

If tools available, verify integration:

```bash
# Git-secrets
git secrets --scan

# Bandit (Python)
bandit -r . -f json

# npm audit (JavaScript)
npm audit --json

# Safety (Python dependencies)
safety check --json
```

---

## Cleanup

```bash
rm -rf /tmp/test-repo-security
```

---

## Test Results

**Date**: _____________
**Executor**: _____________
**Status**: [ ] PASS | [ ] FAIL | [ ] SKIP

**Vulnerabilities Detected**: ____ / 19
**Critical Detected**: ____ / 12
**High Detected**: ____ / 6
**Medium Detected**: ____ / 1

**False Positives**: ____
**False Negatives**: ____

**Notes**:
-
-

**Issues Found**:
-
-
