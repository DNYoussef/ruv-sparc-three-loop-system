# Audit Pipeline - Quick Start

## 🚀 One Command to Production-Ready Code

```bash
/audit-pipeline
```

That's it! This runs all 3 phases automatically:
1. **Theater Detection** - Find all mocks/fakes
2. **Functionality** - Test everything (with Codex auto-fix)
3. **Style** - Polish to production standards

## 📊 The 3 Phases

| Phase | What It Does | Time | Can Run Alone |
|-------|--------------|------|---------------|
| 1. Theater | Find mocks, TODOs, stubs | 2-5 min | `/theater-detection-audit` |
| 2. Functionality | Test + Codex auto-fix | 5-15 min | `/functionality-audit` |
| 3. Style | Lint + refactor + polish | 3-10 min | `/style-audit` |

**Total**: ~15-30 min for medium project

## 💡 Common Use Cases

### Pre-Production Deployment
```bash
/audit-pipeline "Pre-production quality gate for v2.0 release"
```

### After Rapid Prototyping
```bash
/audit-pipeline "Harden prototype code for production"
```

### Legacy Code Cleanup
```bash
/audit-pipeline "Modernize legacy authentication module"
```

### Before Code Review
```bash
/audit-pipeline "Clean up feature branch before PR"
```

## 🎯 What You Get

### Before
```python
def get_user(id):
    return {"id": 123, "name": "Test"}  # FAKE DATA - TODO: real DB
```

### After Phase 1 (Theater Detected)
```
⚠️ Theater found: Hardcoded test data in get_user()
→ Needs: Real database query implementation
```

### After Phase 2 (Functionality + Codex)
```python
def get_user(user_id):
    with get_db_connection() as conn:
        result = conn.execute("SELECT * FROM users WHERE id=?", (user_id,))
        if not result:
            raise UserNotFoundError(f"User {user_id} not found")
        return {"id": result[0], "name": result[1]}
```
✅ Tested in sandbox, Codex fixed 2 edge cases

### After Phase 3 (Style Polish)
```python
def get_user(user_id: int) -> Dict[str, Any]:
    """
    Fetch user from database by ID.

    Args:
        user_id: Unique user identifier

    Returns:
        User data dictionary

    Raises:
        UserNotFoundError: If user doesn't exist
    """
    logger.debug(f"Fetching user {user_id}")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, email FROM users WHERE id = ?",
            (user_id,)
        )
        result = cursor.fetchone()

        if not result:
            raise UserNotFoundError(f"User {user_id} not found")

        return {
            "id": result[0],
            "name": result[1],
            "email": result[2]
        }
```
✅ Production-ready with types, docs, logging, validation!

## 🤖 Codex Integration (Phase 2)

The functionality phase uses **Codex's Full Auto mode** in a sandbox:

```
For each failing test:
1. Capture error
2. Spawn: codex --full-auto "Fix this test failure..."
3. Codex fixes autonomously in safe sandbox
4. Re-test
5. If still failing → iterate (max 5 times)
6. If passing → apply fix to main code
```

**Safe because**:
- Sandbox isolated (network disabled, CWD only)
- Changes tested before applying
- Regression checks prevent breaking other code
- You approve final results

## ⚙️ Configuration

### Run Specific Phases
```bash
/audit-pipeline --phases=1,2      # Just theater + functionality
/audit-pipeline --phases=3        # Just style (if 1,2 done)
```

### Codex Control
```bash
/audit-pipeline --codex=off       # Manual fixes only
/audit-pipeline --codex=assisted  # Codex suggests, you approve
/audit-pipeline --codex=auto      # Full auto (default)
```

### Strictness
```bash
/audit-pipeline --strict          # Block on any issues
/audit-pipeline --lenient         # Warnings only
```

## 📈 Report

You get a comprehensive report:

```markdown
# Audit Pipeline Report

## Summary
✅ Theater: 12 found, 12 completed
✅ Functionality: 156 tests, all passing (Codex fixed 8)
✅ Style: 67 issues fixed
✅ Quality: 47% → 94%
✅ Production Ready: APPROVED

## Phase Details
[Theater report]
[Functionality report with Codex iterations]
[Style report]

## Metrics
- Test coverage: 65% → 91%
- Maintainability: C → A
- Technical debt: 8 weeks → 3 days
```

## ⏱️ Time Estimates

- **Small** (< 1K lines): 5-10 min
- **Medium** (1K-10K): 15-30 min
- **Large** (10K-50K): 30-60 min
- **Huge** (50K+): 1-3 hours

## ✅ Success Checklist

After pipeline completes:
- [ ] Review audit report
- [ ] Check all tests passing
- [ ] Validate critical changes
- [ ] Run manual smoke test
- [ ] Commit with detailed message

## 🔗 Individual Skills

If you need just one phase:

```bash
/theater-detection-audit   # Phase 1 alone
/functionality-audit       # Phase 2 alone
/style-audit               # Phase 3 alone
```

## 🎓 Best Practices

1. **Commit first** - So you can rollback if needed
2. **Trust Codex** - It's good at fixing routine test failures
3. **Review report** - Understand what changed
4. **Run before deploy** - Catch issues early
5. **Use regularly** - Maintain quality continuously

## 🆘 Troubleshooting

**"Pipeline taking too long"**
→ Run phases individually, focus on critical code first

**"Codex can't fix a test"**
→ It escalates to you after 5 tries for manual fix

**"Style changes break tests"**
→ Pipeline auto-detects and rolls back, then retries

**"Too many issues found"**
→ Use `--lenient` for first pass, then `--strict` for cleanup

---

**Bottom line**: Run `/audit-pipeline` before every production deployment. It transforms code from "works on my machine" to "production-ready" automatically!

See `docs/agents/audit-pipeline-guide.md` for full documentation.
