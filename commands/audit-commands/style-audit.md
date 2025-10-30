---
name: style-audit
binding: micro-skill:style-audit
category: audit
version: 1.0.0
---

# /style-audit

Audits code against style rules, quality guidelines, and best practices, then rewrites to meet standards.

## Usage
```bash
/style-audit <path> [options]
```

## Parameters
- `path` - File or directory to audit (required)
- `--rules` - Style rules file (default: .eslintrc, .prettierrc, etc.)
- `--fix` - Auto-fix style issues (default: true)
- `--output` - Output file for report (default: stdout)

## Examples
```bash
# Audit and auto-fix
/style-audit src/

# Audit only (no fixes)
/style-audit src/ --fix false

# Custom rules
/style-audit src/ --rules custom-style-rules.json --output style-report.json
```

## What It Checks
- Linting rules enforcement
- Naming convention compliance
- Code organization and structure
- Readability and maintainability
- Documentation completeness

## Chains with
- `/theater-detect` → `/functionality-audit` → `/style-audit`
- Part of `/audit-pipeline` cascade

## See also
- `/functionality-audit` - Verify functionality first
- `/audit-pipeline` - Run all audits
- `/lint-code` - Quick linting only
