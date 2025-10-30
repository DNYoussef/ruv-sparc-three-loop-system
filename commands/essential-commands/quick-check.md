---
name: quick-check
binding: skill:quick-quality-check
category: essential
version: 1.0.0
---

# /quick-check

Lightning-fast quality check using parallel execution (< 30 seconds).

## Usage
```bash
/quick-check <path>
```

## Parameters
- `path` - File or directory to check (required)
- `--detailed` - Show detailed output (default: false)

## What It Does

Runs 5 quality checks **in parallel**:
1. 🎭 Theater detection (mocks, TODOs)
2. 🎨 Style audit (linting, formatting)
3. 🔒 Security scan (vulnerabilities)
4. 🧪 Test execution (basic tests)
5. 📊 Token usage analysis

**Result**: Quality score (0-100) + prioritized issues

## Examples

```bash
# Quick check current directory
/quick-check .

# Check specific file
/quick-check src/api/users.js

# Detailed output
/quick-check src/ --detailed
```

## Output

```
🚀 Quick Quality Check Complete (23s)

Quality Score: 87/100

Issues Found:
  🚨 Critical (0)
  ⚠️  High (2)
  📝 Medium (5)
  💡 Low (8)

Next Steps:
  - Fix 2 high-priority issues
  - Review 5 medium-priority items
```

## Chains With

```bash
# Quick check → full audit if score < 80
/quick-check src/ && echo "✅ Good" || /audit-pipeline src/

# Quick check before commit
/quick-check . && git commit
```

## See Also
- `/audit-pipeline` - Full 3-phase audit
- `/production-readiness` - Pre-deployment check
- `/code-review` - PR review
