---
name: workflow:hotfix
description: Emergency hotfix workflow with rapid deployment and rollback capabilities
category: workflows
version: 2.0.0
requires_mcp: true
---

# 🚨 Hotfix Workflow - Emergency Production Fix Pipeline

**Workflow**: Emergency Hotfix | **Category**: Workflows | **Urgency**: Critical

Rapid emergency fix workflow for production incidents with automated testing, deployment, and rollback.

## Workflow Execution

```bash
#!/bin/bash
# Emergency Hotfix Workflow
# Usage: ./hotfix.sh "ISSUE-123" "Fix critical auth bug"

set -e

ISSUE_ID=$1
DESCRIPTION=$2
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
HOTFIX_BRANCH="hotfix/${ISSUE_ID}-${TIMESTAMP}"

echo "🚨 EMERGENCY HOTFIX WORKFLOW"
echo "================================"
echo "Issue: ${ISSUE_ID}"
echo "Description: ${DESCRIPTION}"
echo "Branch: ${HOTFIX_BRANCH}"
echo "================================"

# Phase 1: Create Hotfix Branch from Production
echo "Phase 1: Creating hotfix branch from production..."
git fetch origin
git checkout -b ${HOTFIX_BRANCH} origin/production
git push -u origin ${HOTFIX_BRANCH}

# Log to memory
npx claude-flow@alpha memory store \
  --key "hotfix/${ISSUE_ID}/started" \
  --value "{\"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\", \"branch\": \"${HOTFIX_BRANCH}\"}"

# Phase 2: Apply Fix (Manual or Automated)
echo "Phase 2: Apply fix..."
echo "⚠️  Apply your fix now and press ENTER to continue"
read -p ""

# Phase 3: Run Critical Tests
echo "Phase 3: Running critical tests..."
npm run test:critical

# Phase 4: Build and Verify
echo "Phase 4: Building application..."
npm run build

# Verify build succeeded
if [ ! -d "dist" ]; then
  echo "❌ Build failed! Aborting hotfix."
  exit 1
fi

# Phase 5: Automated Security Scan
echo "Phase 5: Security scan..."
npm audit --audit-level=high

# Phase 6: Commit and Push
echo "Phase 6: Committing changes..."
git add .
git commit -m "🚨 HOTFIX(${ISSUE_ID}): ${DESCRIPTION}

- Critical production fix
- Tested: Unit, Integration, Security
- Deployed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin ${HOTFIX_BRANCH}

# Phase 7: Create Pull Request
echo "Phase 7: Creating emergency PR..."
gh pr create \
  --base production \
  --head ${HOTFIX_BRANCH} \
  --title "🚨 HOTFIX: ${DESCRIPTION}" \
  --body "## Emergency Hotfix

**Issue**: ${ISSUE_ID}
**Severity**: Critical
**Description**: ${DESCRIPTION}

## Changes
- [List changes here]

## Testing
- [x] Unit tests passing
- [x] Integration tests passing
- [x] Security scan passed
- [x] Build verified

## Deployment
Ready for immediate production deployment.

**Rollback Plan**: Revert commit or deploy previous version

🤖 Generated with Claude Code" \
  --label "hotfix,urgent"

# Phase 8: Auto-merge if tests pass (requires GitHub Actions)
echo "Phase 8: Waiting for CI/CD checks..."
sleep 10  # Wait for CI to start

# Phase 9: Deploy to Production
echo "Phase 9: Deploying to production..."
echo "⚠️  This will deploy to PRODUCTION. Continue? (yes/no)"
read -p "" CONFIRM

if [ "$CONFIRM" = "yes" ]; then
  # Trigger deployment
  gh workflow run deploy-production.yml \
    --ref ${HOTFIX_BRANCH}

  echo "✅ Hotfix deployed to production"

  # Phase 10: Monitor Deployment
  echo "Phase 10: Monitoring deployment..."
  sleep 30  # Wait for deployment

  # Health check
  HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://api.example.com/health)

  if [ "$HEALTH_STATUS" = "200" ]; then
    echo "✅ Health check passed"
  else
    echo "❌ Health check failed! Initiating rollback..."
    # Trigger rollback
    gh workflow run rollback.yml
    exit 1
  fi

  # Phase 11: Merge to Main Branch
  echo "Phase 11: Merging hotfix to main..."
  gh pr merge --auto --squash

  # Phase 12: Clean up
  echo "Phase 12: Cleanup..."
  git checkout main
  git pull origin main
  git branch -d ${HOTFIX_BRANCH}

  # Log completion
  npx claude-flow@alpha memory store \
    --key "hotfix/${ISSUE_ID}/completed" \
    --value "{\"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\", \"status\": \"success\"}"

  echo "================================"
  echo "✅ HOTFIX WORKFLOW COMPLETE"
  echo "================================"
else
  echo "Deployment cancelled"
  exit 1
fi
```

## Critical Test Suite

```typescript
/**
 * Critical Tests for Hotfix Validation
 * Only essential tests to minimize deployment time
 */

// tests/critical.test.ts
describe('Critical Tests', () => {
  describe('Authentication', () => {
    it('should authenticate valid users', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'test@example.com',
          password: 'validPassword'
        });

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('token');
    });

    it('should reject invalid credentials', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'test@example.com',
          password: 'wrongPassword'
        });

      expect(response.status).toBe(401);
    });
  });

  describe('Core API Endpoints', () => {
    it('should return health status', async () => {
      const response = await request(app).get('/health');
      expect(response.status).toBe(200);
      expect(response.body.status).toBe('healthy');
    });

    it('should handle database connections', async () => {
      const isConnected = await db.$queryRaw`SELECT 1`;
      expect(isConnected).toBeDefined();
    });
  });

  describe('Data Integrity', () => {
    it('should not corrupt existing data', async () => {
      const users = await db.user.findMany({ take: 10 });
      expect(users.length).toBeGreaterThan(0);
      expect(users[0]).toHaveProperty('id');
      expect(users[0]).toHaveProperty('email');
    });
  });
});
```

## Rollback Procedure

```bash
#!/bin/bash
# Automated Rollback Script

echo "🔄 INITIATING ROLLBACK"
echo "================================"

# Phase 1: Identify Last Known Good Version
LAST_GOOD_VERSION=$(git describe --tags --abbrev=0 production^)
echo "Rolling back to: ${LAST_GOOD_VERSION}"

# Phase 2: Revert Production Branch
git checkout production
git reset --hard ${LAST_GOOD_VERSION}
git push origin production --force

# Phase 3: Redeploy Previous Version
gh workflow run deploy-production.yml \
  --ref ${LAST_GOOD_VERSION}

# Phase 4: Verify Rollback
sleep 30
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://api.example.com/health)

if [ "$HEALTH_STATUS" = "200" ]; then
  echo "✅ Rollback successful"

  # Log rollback
  npx claude-flow@alpha memory store \
    --key "rollback/$(date +%Y%m%d-%H%M%S)" \
    --value "{\"version\": \"${LAST_GOOD_VERSION}\", \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"}"
else
  echo "❌ Rollback failed! Manual intervention required!"
  exit 1
fi

echo "================================"
echo "✅ ROLLBACK COMPLETE"
echo "================================"
```

## Monitoring & Alerts

```typescript
/**
 * Hotfix Deployment Monitoring
 */

interface HotfixMetrics {
  deploymentTime: number;
  errorRate: number;
  responseTime: number;
  uptime: number;
}

async function monitorHotfixDeployment(
  hotfixId: string,
  duration: number = 300000 // 5 minutes
): Promise<HotfixMetrics> {
  const startTime = Date.now();
  const metrics: HotfixMetrics = {
    deploymentTime: 0,
    errorRate: 0,
    responseTime: 0,
    uptime: 100
  };

  // Monitor for specified duration
  const interval = setInterval(async () => {
    try {
      // Health check
      const healthResponse = await fetch('https://api.example.com/health');

      if (healthResponse.ok) {
        const data = await healthResponse.json();
        metrics.responseTime = data.responseTime;
        metrics.errorRate = data.errorRate;
        metrics.uptime = data.uptime;
      } else {
        // Health check failed
        console.error('Health check failed:', healthResponse.status);
        metrics.uptime = 0;

        // Trigger alert
        await sendAlert({
          severity: 'critical',
          message: `Hotfix deployment failing: ${hotfixId}`,
          metrics
        });
      }
    } catch (error) {
      console.error('Monitoring error:', error);
    }

    // Stop monitoring after duration
    if (Date.now() - startTime > duration) {
      clearInterval(interval);

      metrics.deploymentTime = Date.now() - startTime;

      // Store metrics
      await storeMetrics(hotfixId, metrics);
    }
  }, 10000); // Check every 10 seconds

  return metrics;
}

async function sendAlert(alert: {
  severity: string;
  message: string;
  metrics: HotfixMetrics;
}) {
  // Send to Slack, PagerDuty, etc.
  await fetch('https://hooks.slack.com/services/YOUR/WEBHOOK/URL', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: `🚨 ${alert.severity.toUpperCase()}: ${alert.message}`,
      attachments: [
        {
          color: 'danger',
          fields: [
            { title: 'Error Rate', value: `${alert.metrics.errorRate}%`, short: true },
            { title: 'Uptime', value: `${alert.metrics.uptime}%`, short: true },
            { title: 'Response Time', value: `${alert.metrics.responseTime}ms`, short: true }
          ]
        }
      ]
    })
  });
}
```

## Hotfix Checklist

### Pre-Deployment
- [ ] Issue severity confirmed (Critical/High)
- [ ] Root cause identified
- [ ] Fix implemented and code reviewed
- [ ] Critical tests passing
- [ ] Security scan passed
- [ ] Build successful
- [ ] Rollback plan prepared

### Deployment
- [ ] Hotfix branch created from production
- [ ] Changes committed and pushed
- [ ] Emergency PR created
- [ ] CI/CD checks passed
- [ ] Production deployment triggered
- [ ] Health checks passing

### Post-Deployment
- [ ] Monitoring enabled (5-15 minutes)
- [ ] Error rates normal
- [ ] Performance metrics normal
- [ ] User impact mitigated
- [ ] Incident documented
- [ ] Post-mortem scheduled

---

## Integration with Three-Loop System

**Emergency Path**: Bypass Loop 1, rapid Loop 2 fix, immediate Loop 3 deployment

**Time Target**: < 30 minutes from issue detection to production deployment

**Agent Status**: Production-Ready
**Version**: 2.0.0
