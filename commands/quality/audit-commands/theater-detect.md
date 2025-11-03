---
name: theater-detect
binding: micro-skill:theater-detection-audit
category: audit
version: 1.0.0
---

# /theater-detect

Detects placeholder code, mock data, TODO markers, and incomplete implementations.

## Usage
```bash
/theater-detect <path> [options]
```

## Parameters
- `path` - File or directory to scan (required)
- `--output` - Output file for report (default: stdout)
- `--format` - Report format: json|markdown|text (default: markdown)
- `--fix` - Auto-complete detected theater with production code

## Examples
```bash
# Scan directory
/theater-detect src/

# Generate JSON report
/theater-detect src/ --output theater-report.json --format json

# Auto-fix theater code
/theater-detect src/ --fix
```

## Chains with
- `/theater-detect` → `/functionality-audit` → `/style-audit`
- Part of `/audit-pipeline` cascade

## See also
- `/functionality-audit` - Verify code works
- `/style-audit` - Polish code quality
- `/audit-pipeline` - Run all audits
