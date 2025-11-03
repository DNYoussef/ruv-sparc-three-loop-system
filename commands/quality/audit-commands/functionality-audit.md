---
name: functionality-audit
binding: micro-skill:functionality-audit
category: audit
version: 1.0.0
---

# /functionality-audit

Validates that code actually works through sandbox testing with Codex auto-fix iteration.

## Usage
```bash
/functionality-audit <path> [options]
```

## Parameters
- `path` - File or directory to test (required)
- `--model` - AI model for auto-fix: codex-auto|claude (default: codex-auto)
- `--max-iterations` - Max auto-fix iterations per test (default: 5)
- `--sandbox` - Enable isolated sandbox testing (default: true)
- `--output` - Output file for report (default: stdout)

## Examples
```bash
# Test with Codex auto-fix
/functionality-audit src/ --model codex-auto

# Custom iteration limit
/functionality-audit src/ --max-iterations 10

# Generate report
/functionality-audit src/ --output functionality-report.json
```

## How It Works
1. Execute test suite in sandbox
2. For each failure:
   - Spawn Codex in isolated sandbox
   - Auto-fix the failing test
   - Re-run test
   - Iterate up to 5 times
   - Apply fix if test passes
3. Escalate to user if still failing

## Chains with
- `/theater-detect` → `/functionality-audit` → `/style-audit`
- Part of `/audit-pipeline` cascade

## See also
- `/theater-detect` - Find placeholder code first
- `/style-audit` - Polish after fixing functionality
- `/audit-pipeline` - Run all audits
- `/codex-auto` - Direct Codex access
