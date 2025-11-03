# Deployment Checklist

**Feature**: [Feature Name]
**Version**: [Version Number]
**Deployment Date**: [YYYY-MM-DD]
**Deployer**: [Your Name]

---

## Pre-Deployment Checklist

### Code Quality
- [ ] All code reviewed and approved
- [ ] No merge conflicts
- [ ] Code formatted and linted
- [ ] No console.log or debug statements
- [ ] Comments and documentation updated
- [ ] Version numbers updated

### Testing
- [ ] All unit tests passing (≥80% coverage)
- [ ] All integration tests passing
- [ ] All E2E tests passing
- [ ] Performance tests meet benchmarks
- [ ] Security scan completed (0 critical issues)
- [ ] Accessibility tests passing (WCAG 2.1 AA)
- [ ] Cross-browser testing completed
- [ ] Mobile responsiveness verified

### Dependencies
- [ ] All dependencies up to date
- [ ] No critical security vulnerabilities in dependencies
- [ ] Package-lock.json / yarn.lock committed
- [ ] Build succeeds without warnings

### Configuration
- [ ] Environment variables configured
- [ ] API keys and secrets in secure storage
- [ ] Database migrations prepared
- [ ] Feature flags configured
- [ ] CDN/cache settings reviewed
- [ ] CORS policies verified

### Documentation
- [ ] README updated
- [ ] API documentation updated
- [ ] User documentation updated
- [ ] CHANGELOG updated
- [ ] Architecture diagrams updated
- [ ] Runbook/troubleshooting guide prepared

---

## Deployment Process

### Step 1: Pre-Deployment Communication
- [ ] Notify stakeholders of deployment window
- [ ] Post deployment notice in team channels
- [ ] Schedule deployment during low-traffic window
- [ ] Ensure on-call engineer is available

### Step 2: Database Migrations (if applicable)
- [ ] Backup production database
- [ ] Run migrations in staging first
- [ ] Verify migration success in staging
- [ ] Prepare rollback scripts
- [ ] Run production migrations
- [ ] Verify migration success in production

### Step 3: Code Deployment
- [ ] Merge to main/master branch
- [ ] Tag release in git
- [ ] Trigger CI/CD pipeline
- [ ] Monitor build progress
- [ ] Verify build artifacts

### Step 4: Gradual Rollout
- [ ] Deploy to canary (5% traffic)
- [ ] Monitor canary metrics for 30 minutes
- [ ] Deploy to 25% traffic
- [ ] Monitor metrics for 30 minutes
- [ ] Deploy to 50% traffic
- [ ] Monitor metrics for 30 minutes
- [ ] Deploy to 100% traffic

### Step 5: Post-Deployment Verification
- [ ] Run smoke tests in production
- [ ] Verify critical user flows
- [ ] Check application logs for errors
- [ ] Verify monitoring dashboards
- [ ] Test feature flags
- [ ] Verify API responses
- [ ] Check database connections

---

## Monitoring & Validation

### Performance Metrics
- [ ] Response times within acceptable range (<200ms p95)
- [ ] Error rates below threshold (<0.1%)
- [ ] CPU usage normal (<70%)
- [ ] Memory usage normal (<80%)
- [ ] Database query performance acceptable

### Business Metrics
- [ ] User adoption tracking configured
- [ ] Conversion funnel tracking working
- [ ] Analytics events firing correctly
- [ ] A/B test tracking (if applicable)

### Health Checks
- [ ] Application health endpoint responding
- [ ] Load balancer health checks passing
- [ ] Database connections healthy
- [ ] External API integrations working
- [ ] Background jobs processing

---

## Rollback Plan

### Rollback Triggers
Initiate rollback if:
- Error rate exceeds 1%
- Response time p95 exceeds 500ms
- Critical functionality broken
- Data corruption detected
- Security vulnerability exposed

### Rollback Steps
1. [ ] Stop deployment immediately
2. [ ] Revert to previous git tag
3. [ ] Trigger rollback pipeline
4. [ ] Rollback database migrations (if needed)
5. [ ] Verify rollback success
6. [ ] Notify stakeholders
7. [ ] Document issues for post-mortem

---

## Post-Deployment

### Immediate (Within 1 Hour)
- [ ] Monitor error logs
- [ ] Check performance dashboards
- [ ] Verify user reports/feedback
- [ ] Ensure all team members aware of deployment status

### Short-term (Within 24 Hours)
- [ ] Review deployment metrics
- [ ] Analyze user adoption
- [ ] Address any minor issues
- [ ] Update stakeholders with deployment summary

### Long-term (Within 1 Week)
- [ ] Conduct post-deployment review
- [ ] Document lessons learned
- [ ] Update deployment procedures
- [ ] Archive deployment artifacts

---

## Communication Templates

### Pre-Deployment Notice
```
🚀 Deployment Scheduled

Feature: [Feature Name]
Date: [Date]
Time: [Time] UTC
Duration: ~[Duration]
Impact: [Expected impact]

We will be deploying [feature description]. Users may experience [potential disruptions].

Questions? Contact: [Your Name/Team]
```

### Deployment Complete
```
✅ Deployment Complete

Feature: [Feature Name]
Deployed: [Timestamp]
Status: Success
Rollout: 100%

All systems nominal. Monitoring continues for 24 hours.

Metrics:
- Response time: [XX]ms (p95)
- Error rate: [X]%
- Tests passing: [XX]%

Report issues to: [Contact]
```

### Rollback Notice
```
⚠️ Deployment Rollback

Feature: [Feature Name]
Rolled Back: [Timestamp]
Reason: [Brief reason]

We have rolled back to the previous version due to [issue]. Service is now stable.

Post-mortem scheduled for: [Date/Time]

Contact: [Your Name/Team]
```

---

## Incident Response

### Severity Levels
- **P0 (Critical)**: Complete outage, data loss, security breach
- **P1 (High)**: Major feature down, significant performance degradation
- **P2 (Medium)**: Minor feature issue, workaround available
- **P3 (Low)**: Cosmetic issue, minimal impact

### Escalation Path
1. **First Responder**: [Name/Role]
2. **Tech Lead**: [Name]
3. **Engineering Manager**: [Name]
4. **On-Call Engineer**: [Contact info]

### Response Times
- **P0**: Immediate response, rollback within 15 minutes
- **P1**: Response within 30 minutes
- **P2**: Response within 2 hours
- **P3**: Response within 24 hours

---

## Sign-Off

### Pre-Deployment Approval
| Role | Name | Approved | Date |
|------|------|----------|------|
| Tech Lead | | [ ] | |
| QA Lead | | [ ] | |
| Product Manager | | [ ] | |
| Security Lead | | [ ] | |

### Post-Deployment Verification
| Role | Name | Verified | Date |
|------|------|----------|------|
| Deployer | | [ ] | |
| QA Engineer | | [ ] | |
| On-Call Engineer | | [ ] | |

---

## Notes

[Add any deployment-specific notes, special considerations, or important context here]

---

**Deployment Log**

| Timestamp | Action | Result | Notes |
|-----------|--------|--------|-------|
| [Time] | [Action taken] | [Success/Fail] | [Additional context] |
| [Time] | [Action taken] | [Success/Fail] | [Additional context] |
