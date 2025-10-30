---
name: deploy-check
binding: skill:production-readiness
category: essential
version: 1.0.0
---

# /deploy-check

Comprehensive pre-deployment validation with quality gates.

## Usage
```bash
/deploy-check <path> [environment]
```

## Parameters
- `path` - Directory to validate (required)
- `environment` - staging|production (default: production)
- `--skip-performance` - Skip performance tests (default: false)
- `--strict-mode` - Enforce strict quality gates (default: true)

## What It Does

**10-Gate Production Readiness Check**:
1. ✅ **Tests Passing** - 100% test pass rate
2. ✅ **Code Quality** - Score ≥ 85/100
3. ✅ **Test Coverage** - Coverage ≥ 80%
4. ✅ **Security** - Zero critical/high vulnerabilities
5. ✅ **Performance** - Within SLA (avg < 200ms, p95 < 500ms)
6. ✅ **Documentation** - README, deployment docs, rollback plan
7. ✅ **Dependencies** - No vulnerable packages
8. ✅ **Configuration** - Secrets in env vars, not hardcoded
9. ✅ **Monitoring** - Logging, metrics, alerts configured
10. ✅ **Rollback Plan** - Documented and tested

**All gates must pass** for production deployment approval.

## Examples

```bash
# Check production readiness
/deploy-check . production

# Check staging
/deploy-check ./dist staging

# Skip performance tests
/deploy-check . production --skip-performance

# Non-strict mode (warnings instead of errors)
/deploy-check . staging --strict-mode false
```

## Output

```
🚀 Production Readiness Assessment

Environment: production
Gates Passed: 10/10 ✅

Quality Gates:
  ✅ Tests: All passing
  ✅ Quality: 92/100
  ✅ Coverage: 87%
  ✅ Security: Clean
  ✅ Performance: Within SLAs
  ✅ Documentation: Complete

🎉 READY FOR DEPLOYMENT!

Next Steps:
  1. Review deployment checklist
  2. Get required sign-offs
  3. Schedule deployment window
  4. Execute deployment

Deployment Checklist: production-readiness-*/DEPLOYMENT-CHECKLIST.md
```

## Deployment Checklist

Generates comprehensive checklist including:
- Pre-deployment tasks
- Deployment steps
- Post-deployment verification
- Rollback procedure
- Sign-off requirements

## Quality Gates (Production)

| Gate | Requirement | Blocking |
|------|-------------|----------|
| Tests | 100% passing | Yes |
| Quality | ≥ 85/100 | Yes |
| Coverage | ≥ 80% | Yes |
| Security | 0 critical/high | Yes |
| Performance | Within SLAs | Yes |
| Docs | Complete | Warning |
| Dependencies | No vulnerabilities | Warning |
| Configuration | Proper secrets mgmt | Yes |
| Monitoring | Configured | Warning |
| Rollback | Documented | Yes |

## Chains With

```bash
# Check → deploy if ready
/deploy-check . production && /deploy

# Check → fix issues → recheck
/deploy-check . || /fix-issues && /deploy-check .

# Staging → production pipeline
/deploy-check . staging && \
  /deploy staging && \
  /smoke-tests && \
  /deploy-check . production && \
  /deploy production
```

## See Also
- `/audit-pipeline` - Quality audit
- `/quick-check` - Fast quality check
- `/security-scan` - Security only
