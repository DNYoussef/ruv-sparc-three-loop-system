# Secrets Management - 12-Factor App Compliance

## Overview

The Secrets Redaction system prevents API keys, tokens, passwords, and other sensitive credentials from being stored in the memory system. This implements **Factor III: Config** of the 12-Factor App methodology, which mandates storing secrets in environment variables rather than in code or configuration files.

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Store Operation                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Pre-Memory-Store Hook (Interceptor)            │
│  - Intercepts all mcp__claude-flow__memory_store calls      │
│  - Validates key-value pairs before storage                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Secrets Redaction Engine                   │
│  - Loads patterns from secrets-patterns.json                │
│  - Scans value against 20+ secret detection patterns        │
│  - Checks whitelist (test keys, localhost, etc.)           │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ┌────┴────┐
                    │         │
              Secrets    No Secrets
              Detected    Detected
                    │         │
                    ▼         ▼
           ┌────────────┐  ┌────────────┐
           │   BLOCK    │  │   ALLOW    │
           │  Storage   │  │  Storage   │
           └────┬───────┘  └────────────┘
                │
                ▼
         ┌──────────────┐
         │ Log Violation│
         │ Return Error │
         └──────────────┘
```

### Detection Patterns

The system uses regex patterns to detect 20+ types of secrets:

**API Keys:**
- Anthropic API keys (sk-ant-*)
- OpenAI API keys (sk-*)
- Generic API keys

**Version Control Tokens:**
- GitHub Personal Access Tokens (ghp_*)
- GitHub OAuth tokens (gho_*)
- GitHub App tokens (ghs_*)

**Cloud Provider Credentials:**
- AWS Access Keys (AKIA*)
- AWS Secret Keys
- Azure connection strings
- GCP service account JSON

**Third-Party Services:**
- Slack webhooks and tokens
- Stripe API keys
- Twilio API keys

**Authentication:**
- JWT tokens
- Basic Auth headers
- Bearer tokens
- Passwords in plain text

**Infrastructure:**
- Private keys (RSA, EC, SSH)
- Database connection strings
- Redis URLs

### Performance

- **Scan time**: <10ms for typical memory operations
- **Pattern matching**: Compiled regex for maximum performance
- **Size limit**: 1MB per value (configurable)
- **Timeout protection**: Warns if scan exceeds 10ms

## Configuration

### Pattern Configuration

Edit `hooks/12fa/secrets-patterns.json`:

```json
{
  "patterns": [
    {
      "name": "anthropic_api_key",
      "regex": "sk-ant-[a-zA-Z0-9_-]{95,}",
      "severity": "critical",
      "description": "Anthropic API key detected",
      "recommendation": "Use ANTHROPIC_API_KEY environment variable"
    }
  ],
  "whitelisted_patterns": [
    "example\\.com",
    "test[_-]?(key|token|secret)"
  ],
  "config": {
    "block_on_detection": true,
    "redact_in_logs": true,
    "audit_violations": true,
    "max_scan_size": 1048576,
    "performance_timeout_ms": 10
  }
}
```

### Adding Custom Patterns

1. **Via JSON file** (recommended for permanent patterns):

```json
{
  "patterns": [
    {
      "name": "my_custom_secret",
      "regex": "CUSTOM-[A-Z0-9]{16}",
      "severity": "high",
      "description": "Custom secret format",
      "recommendation": "Use CUSTOM_SECRET environment variable"
    }
  ]
}
```

2. **Via code** (for runtime patterns):

```javascript
const { getRedactorInstance } = require('./hooks/12fa/secrets-redaction');

const redactor = getRedactorInstance();
redactor.addCustomPattern({
  name: 'custom_secret',
  regex: 'CUSTOM-[A-Z0-9]{16}',
  severity: 'high',
  description: 'Custom secret format'
});
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `block_on_detection` | boolean | `true` | Block storage when secrets detected |
| `redact_in_logs` | boolean | `true` | Redact secrets in log files |
| `audit_violations` | boolean | `true` | Log violations to audit trail |
| `max_scan_size` | number | `1048576` | Maximum bytes to scan (1MB) |
| `performance_timeout_ms` | number | `10` | Performance warning threshold |

## Usage

### Automatic Protection

All memory store operations are automatically protected:

```javascript
// This will be blocked if it contains secrets
await mcp__claude-flow__memory_store({
  key: "config/api",
  value: JSON.stringify({
    apiKey: "sk-ant-api03-..." // ❌ BLOCKED
  })
});

// This will succeed
await mcp__claude-flow__memory_store({
  key: "config/api",
  value: JSON.stringify({
    apiUrl: "https://api.example.com" // ✅ ALLOWED
  })
});
```

### Manual Validation

Use the hook directly for manual validation:

```bash
# Validate a key-value pair
node hooks/12fa/pre-memory-store.hook.js validate "test/key" "sk-ant-api03-..."

# Get statistics
node hooks/12fa/pre-memory-store.hook.js stats

# Run validation tests
node hooks/12fa/pre-memory-store.hook.js test
```

### Integration in Code

```javascript
const { validateNoSecrets, scanForSecrets } = require('./hooks/12fa/secrets-redaction');

// Validate before storage
try {
  validateNoSecrets('config/api', apiKey);
  // Proceed with storage
} catch (error) {
  console.error('Secret detected:', error.message);
  // Handle error
}

// Scan for secrets without throwing
const detections = scanForSecrets(value);
if (detections.length > 0) {
  console.log('Found secrets:', detections);
}
```

## Best Practices

### 1. Use Environment Variables

**❌ Wrong:**
```javascript
const config = {
  apiKey: "sk-ant-api03-real-key-here"
};
```

**✅ Correct:**
```javascript
const config = {
  apiKey: process.env.ANTHROPIC_API_KEY
};
```

### 2. Use .env Files (Not Committed)

Create `.env` file (add to `.gitignore`):
```bash
ANTHROPIC_API_KEY=sk-ant-api03-...
GITHUB_TOKEN=ghp_...
DATABASE_URL=postgres://user:pass@host/db
```

Load in your application:
```javascript
require('dotenv').config();
const apiKey = process.env.ANTHROPIC_API_KEY;
```

### 3. Use Secret Management Services

For production:
- **AWS Secrets Manager**
- **Azure Key Vault**
- **GCP Secret Manager**
- **HashiCorp Vault**

### 4. Rotate Secrets Regularly

```bash
# Generate new API key
# Update environment variable
# Revoke old key
```

### 5. Never Commit Secrets to Git

```bash
# Check repository for secrets before pushing
git log -p | grep -i "api.key\|password\|secret"

# Use git-secrets to prevent commits
brew install git-secrets
git secrets --install
git secrets --register-aws
```

## Monitoring and Auditing

### View Blocked Attempts

```bash
# Get statistics
node hooks/12fa/pre-memory-store.hook.js stats

# View audit log
cat logs/12fa/secrets-violations.log

# View hook execution log
cat logs/12fa/hook-executions.log
```

### Statistics Format

```json
{
  "totalBlocked": 5,
  "lastBlocked": "2025-11-01T12:34:56.789Z",
  "blockedKeys": {
    "config/api": {
      "count": 3,
      "lastAttempt": "2025-11-01T12:34:56.789Z",
      "lastError": "SECRET DETECTED - Storage blocked..."
    }
  }
}
```

### Violation Log Format

```json
{
  "timestamp": "2025-11-01T12:34:56.789Z",
  "key": "config/api",
  "detections": [
    {
      "name": "anthropic_api_key",
      "severity": "critical",
      "description": "Anthropic API key detected"
    }
  ],
  "blocked": true
}
```

## Testing

### Run Test Suite

```bash
# Run all tests
npm test tests/12fa-compliance/secrets-redaction.test.js

# Run specific test
npm test -- -t "should detect Anthropic API key"

# Run with coverage
npm test -- --coverage tests/12fa-compliance/secrets-redaction.test.js
```

### Manual Testing

```bash
# Run built-in tests
node hooks/12fa/pre-memory-store.hook.js test

# Test specific pattern
node hooks/12fa/pre-memory-store.hook.js validate \
  "test/key" \
  "sk-ant-api03-test-key"
```

### Test Coverage

The test suite validates:
- ✅ All 20+ secret patterns
- ✅ Whitelist functionality
- ✅ Normal data handling
- ✅ Object and array scanning
- ✅ Performance requirements
- ✅ Error messages
- ✅ Edge cases
- ✅ Statistics tracking

## Troubleshooting

### False Positives

If legitimate data is being blocked:

1. Check if it matches a secret pattern
2. Add to whitelist in `secrets-patterns.json`:
   ```json
   {
     "whitelisted_patterns": [
       "your-pattern-here"
     ]
   }
   ```

### Performance Issues

If scans are slow:

1. Check scan time in logs
2. Adjust `max_scan_size` if needed
3. Consider chunking large values
4. Review regex complexity

### Pattern Not Detecting

If a secret type isn't caught:

1. Test the regex pattern
2. Ensure pattern is in `secrets-patterns.json`
3. Reload patterns: `redactor.reloadPatterns()`
4. Check case sensitivity (patterns use `/gi` flags)

## Integration with CI/CD

### Pre-Commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash

# Scan staged files for secrets
git diff --cached --name-only | while read file; do
  if [ -f "$file" ]; then
    node hooks/12fa/pre-memory-store.hook.js validate "ci/check" "$(cat $file)"
  fi
done
```

### CI Pipeline

```yaml
# .github/workflows/security.yml
name: Security Scan
on: [push, pull_request]
jobs:
  secrets-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run secrets scan
        run: node hooks/12fa/pre-memory-store.hook.js test
```

## Support

For issues or questions:
1. Check logs: `logs/12fa/`
2. Run diagnostics: `node hooks/12fa/pre-memory-store.hook.js stats`
3. Review patterns: `hooks/12fa/secrets-patterns.json`
4. Submit issue with violation log

## References

- [12-Factor App - Config](https://12factor.net/config)
- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [AWS Secrets Manager Best Practices](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html)

---

**Version**: 1.0.0
**Last Updated**: 2025-11-01
**Compliance**: 12-Factor App Factor III (Config)
