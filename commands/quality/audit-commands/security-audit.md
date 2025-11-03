---
name: security-audit
category: audit
version: 1.0.0
---

# /security-audit

Comprehensive security scanning and vulnerability detection across codebase.

## Usage
```bash
/security-audit [target_directory] [options]
```

## Parameters
- `target_directory` - Directory to scan (default: current directory)
- `--severity` - Minimum severity level: low|medium|high|critical (default: medium)
- `--fix-auto` - Auto-fix vulnerabilities where possible (default: false)
- `--report-format` - Output format: json|html|markdown (default: markdown)
- `--include-deps` - Include dependency vulnerabilities (default: true)

## What It Does

**Multi-Layer Security Analysis**:
1. 🔍 Static code analysis (SQL injection, XSS, CSRF)
2. 🔐 Secret detection (API keys, tokens, passwords)
3. 📦 Dependency vulnerabilities (npm audit, pip check)
4. 🛡️ Security headers validation
5. 🔒 Authentication/authorization review
6. 🚨 Common vulnerability patterns (OWASP Top 10)
7. 📊 Risk scoring and prioritization
8. 🔧 Auto-fix suggestions

**Detection Categories**:
- **Injection Flaws**: SQL, NoSQL, command injection
- **Authentication Issues**: Weak passwords, session management
- **Sensitive Data Exposure**: Hardcoded secrets, plaintext storage
- **Access Control**: Broken authorization, privilege escalation
- **Security Misconfiguration**: Default configs, verbose errors
- **Cross-Site Scripting (XSS)**: Reflected, stored, DOM-based
- **Insecure Deserialization**: Unsafe object handling
- **Using Components with Known Vulnerabilities**: Outdated dependencies
- **Insufficient Logging**: Missing audit trails
- **Server-Side Request Forgery (SSRF)**: Unvalidated URLs

## Examples

```bash
# Basic security audit
/security-audit

# Audit specific directory with high severity only
/security-audit src/ --severity high

# Full audit with auto-fix
/security-audit --fix-auto true --severity medium

# Generate HTML report
/security-audit --report-format html

# Audit without dependency check
/security-audit --include-deps false
```

## Output

```
🔒 Security Audit Report

Scan Summary:
  Files Scanned: 247
  Issues Found: 12 (3 critical, 5 high, 4 medium)
  Secrets Detected: 2
  Vulnerable Dependencies: 3

Critical Issues:
  ❌ SQL Injection vulnerability in src/auth/login.js:42
     → Use parameterized queries
  ❌ Hardcoded API key in config/api.js:15
     → Move to environment variables
  ❌ Weak password hashing (MD5) in src/users/password.js:28
     → Use bcrypt with salt rounds >= 12

High Severity:
  ⚠️  XSS vulnerability in src/render/template.js:56
  ⚠️  Missing authentication on /admin endpoint
  ⚠️  Outdated crypto library (CVE-2024-1234)
  ⚠️  CSRF token validation missing
  ⚠️  Insecure cookie settings (no httpOnly flag)

Medium Severity:
  ⚡ Verbose error messages expose stack traces
  ⚡ Missing rate limiting on API endpoints
  ⚡ Insufficient input validation
  ⚡ Weak CORS configuration

Dependency Vulnerabilities:
  📦 express@4.16.0 → Upgrade to 4.19.2 (fixes 2 CVEs)
  📦 jsonwebtoken@8.5.0 → Upgrade to 9.0.2 (critical)
  📦 axios@0.21.0 → Upgrade to 1.6.2 (medium)

Risk Score: 73/100 (High Risk)

Auto-Fix Available: 8/12 issues
Run with --fix-auto true to apply fixes

Time: 3.2 seconds
```

## Chains With

```bash
# Security audit → fix → re-audit
/security-audit --fix-auto true && /security-audit

# Audit → dependency update → test
/security-audit && /dependency-audit --update && /regression-test

# Full quality pipeline
/security-audit && /style-audit && /functionality-audit && /production-readiness
```

## See Also
- `/dependency-audit` - Dependency vulnerability scan
- `/functionality-audit` - Code functionality verification
- `/production-readiness` - Pre-deployment validation
- `/sparc:security-review` - Deep security analysis with agent
