---
name: fix-bug
binding: skill:smart-bug-fix
category: essential
version: 1.0.0
---

# /fix-bug

Intelligent bug fixing with RCA + multi-model reasoning + auto-fix + validation.

## Usage
```bash
/fix-bug "<bug description>" <context_path>
```

## Parameters
- `bug_description` - Description of the bug (required)
- `context_path` - Directory or file context (required)
- `--depth` - Investigation depth: shallow|normal|deep (default: deep)
- `--reproduction-steps` - Steps to reproduce (optional)
- `--error-logs` - Path to error logs (optional)

## What It Does

**6-Phase Systematic Debugging**:
1. 🔍 Deep root cause analysis (Claude RCA Agent)
2. 🧠 Context analysis (Gemini MegaContext if large codebase)
3. 💡 Alternative solutions (Multi-model reasoning)
4. 🔧 Implement fix (Codex Auto in sandbox)
5. ✅ Comprehensive testing (Codex iteration, max 5x)
6. 📊 Performance impact analysis

**Models Used**:
- **Claude**: Root cause analysis
- **Gemini**: Large codebase context
- **Codex**: Fix implementation + auto-testing

## Examples

```bash
# Fix bug with description
/fix-bug "API timeout under load" src/api/

# With reproduction steps
/fix-bug "Login fails on Firefox" src/auth/ \
  --reproduction-steps "1. Open Firefox 2. Try login 3. See error"

# With error logs
/fix-bug "Database connection fails" src/db/ \
  --error-logs "logs/error.log"

# Quick shallow fix
/fix-bug "Typo in variable name" src/utils/helper.js --depth shallow
```

## Output

```
🐛 Smart Bug Fix Complete!

Root Cause: Database connection pool exhausted under concurrent requests

Fix Applied:
  - Increased pool size from 10 to 50
  - Added connection timeout handling
  - Implemented connection recycling

Validation:
  ✅ All tests passing
  ✅ No regressions detected
  ✅ Performance improved 23%

Files Changed:
  - src/db/connection.js
  - src/config/database.js
```

## Chains With

```bash
# Fix bug → run full tests → deploy
/fix-bug "bug" src/ && /functionality-audit src/ && /deploy

# Fix multiple bugs in sequence
/fix-bug "bug1" src/ && /fix-bug "bug2" src/ && /audit-pipeline src/
```

## See Also
- `/agent-rca` - RCA only (no fix)
- `/codex-auto` - Direct Codex fix
- `/functionality-audit` - Testing only
