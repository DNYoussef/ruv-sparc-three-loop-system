---
name: workflow:rollback
category: workflow
version: 1.0.0
---

# /workflow:rollback

Intelligent rollback orchestration with automated root cause analysis.

## Usage
```bash
/workflow:rollback [options]
```

## Parameters
- `--environment` - Target environment: dev|staging|production (required)
- `--to` - Target version to rollback to (auto-detect if not specified)
- `--reason` - Rollback reason (required for production)
- `--immediate` - Skip validation gates (default: false)
- `--analyze-failure` - Run root cause analysis (default: true)
- `--notify` - Send rollback notifications (default: true)
- `--preserve-data` - Preserve database state (default: true)
- `--deployment` - Specific deployment name (K8s only)

## What It Does

**Intelligent Rollback Process**:
1. 🔍 **Detect Issue**: Identify deployment problems
2. 📊 **Impact Analysis**: Assess rollback impact
3. 🎯 **Target Selection**: Choose rollback version
4. 🚦 **Validation Gate**: Pre-rollback checks
5. 🔄 **Execute Rollback**: Restore previous version
6. 🏥 **Health Check**: Verify rollback success
7. 🧪 **Smoke Test**: Validate functionality
8. 🔬 **Root Cause Analysis**: Identify failure reason
9. 📢 **Notify**: Send rollback status

**Rollback Strategies**:
- **Instant**: Switch traffic immediately (blue-green)
- **Gradual**: Slowly shift traffic back (canary reverse)
- **Database**: Restore DB to previous state (optional)
- **Full**: Complete environment rollback

## Examples

```bash
# Auto-detect and rollback
/workflow:rollback --environment production

# Rollback to specific version
/workflow:rollback --environment staging --to v1.2.8

# Immediate rollback (skip validation)
/workflow:rollback --environment production --immediate true --reason "Critical bug"

# Rollback with RCA
/workflow:rollback --environment staging --analyze-failure true

# Rollback specific deployment
/workflow:rollback --environment production --deployment myapp-web --to v1.2.9

# Rollback without data preservation
/workflow:rollback --environment dev --preserve-data false
```

## Output

```
🔄 Rollback Workflow Started

Environment: production
Current Version: v1.3.0 (blue)
Rollback Reason: High error rate detected (12.4%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1: Issue Detection & Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔍 Analyzing current deployment...

  Critical Metrics:
    ❌ Error Rate: 12.4% (SLA: <0.1%) - CRITICAL
    ❌ Response Time: 3,245ms (SLA: <100ms) - CRITICAL
    ⚠️  CPU Usage: 98% (SLA: <80%) - HIGH
    ⚠️  Memory Usage: 92% (SLA: <85%) - HIGH
    ❌ Failed Requests: 1,234 in last 5 minutes

  Error Breakdown:
    - 500 Internal Server Error: 834 (67%)
    - 503 Service Unavailable: 289 (23%)
    - 504 Gateway Timeout: 111 (10%)

  🚨 Rollback recommended: Multiple SLA violations
  ⏱️  Duration: 8.3s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 2: Rollback Target Selection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎯 Identifying stable version...

  Recent Deployments:
    v1.3.0 (current): Deployed 12m ago - FAILING ❌
    v1.2.9 (previous): Deployed 2d ago - STABLE ✅
      Error Rate: 0.02%
      Response Time: 84ms
      Uptime: 99.98%
    v1.2.8: Deployed 5d ago - STABLE ✅

  Selected Rollback Target: v1.2.9
  Environment: green (already running, blue-green setup)

  ⏱️  Duration: 2.1s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 3: Pre-Rollback Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ Green environment (v1.2.9): 3/3 pods healthy
  ✅ Database: Compatible with v1.2.9
  ✅ Redis: Connection pool available
  ✅ External services: All reachable
  ✅ Load balancer: Ready for traffic switch

  Impact Assessment:
    - Downtime: 0s (blue-green switch)
    - Data loss: None (preserve-data enabled)
    - User sessions: Preserved
    - In-flight requests: Will complete on blue

  ⏱️  Duration: 4.7s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 4: Execute Rollback
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔀 Switching traffic: blue (v1.3.0) → green (v1.2.9)

  ✅ Service selector updated
  ✅ Ingress backend updated
  ✅ Load balancer reconfigured
  ✅ DNS propagated

  Traffic Routing:
    Blue (v1.3.0): 0%
    Green (v1.2.9): 100% ✅

  ✅ Rollback executed
  ⏱️  Duration: 6.4s (instant traffic switch)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 5: Post-Rollback Health Check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🏥 Monitoring green environment...

  [00:10] Error Rate: 0.02% ✅ | Response Time: 84ms ✅
  [00:20] Error Rate: 0.01% ✅ | Response Time: 82ms ✅
  [00:30] Error Rate: 0.02% ✅ | Response Time: 86ms ✅
  [00:40] Error Rate: 0.00% ✅ | Response Time: 83ms ✅
  [00:50] Error Rate: 0.01% ✅ | Response Time: 85ms ✅
  [01:00] Error Rate: 0.00% ✅ | Response Time: 84ms ✅

  ✅ Metrics stabilized
  ✅ Error rate normalized
  ✅ Response time within SLA

  ⏱️  Duration: 1m 0s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 6: Smoke Test Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🧪 Running smoke tests...

  ✅ User login: Success (78ms)
  ✅ Dashboard load: Success (112ms)
  ✅ API endpoints: 15/15 passed
  ✅ Database queries: All successful
  ✅ External integrations: Working

  ⏱️  Duration: 5.2s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 7: Root Cause Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔬 Analyzing v1.3.0 failure...

  Failure Timeline:
    12:34:22 - Deployment v1.3.0 started
    12:35:08 - All pods healthy
    12:35:45 - First 500 errors detected
    12:36:12 - Error rate spike: 2.1%
    12:37:30 - Error rate critical: 12.4%
    12:38:04 - Rollback initiated

  Root Causes Identified:
    1. ❌ Memory Leak (CRITICAL)
       Location: src/services/cache.service.ts:156
       Issue: Event listeners not cleaned up
       Impact: Memory exhaustion after 2 minutes
       Evidence:
         - Memory growth: 45MB → 892MB in 3 minutes
         - Event listener count: 12,345 (expected: ~50)
         - GC unable to reclaim

    2. ❌ N+1 Database Query (HIGH)
       Location: src/api/users/get-users.controller.ts:42
       Issue: Missing JOIN, executing 1+N queries
       Impact: Database connection pool exhausted
       Evidence:
         - Query count: 1,234 per request (expected: 1)
         - DB connection time: 2,890ms (expected: <50ms)
         - Connection pool: 100/100 used

    3. ⚠️  Missing Error Handling (MEDIUM)
       Location: src/middleware/auth.middleware.ts:67
       Issue: Unhandled promise rejection
       Impact: Process crashes on auth failure
       Evidence:
         - Uncaught exceptions: 234
         - Pod restarts: 12 in 3 minutes

  Recommended Fixes:
    ✅ Add event listener cleanup in componentWillUnmount
    ✅ Implement eager loading with JOIN
    ✅ Add try/catch blocks and error boundaries
    ✅ Add integration test for auth edge cases

  Generated Report: /tmp/rca-v1.3.0-failure.md

  ⏱️  Duration: 18.7s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 8: Cleanup Failed Deployment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🧹 Cleaning up blue environment (v1.3.0)...

  ✅ Blue deployment scaled to 0
  ✅ Failed pods terminated
  ✅ Resources released
  ✅ Image retained (for debugging)
  ✅ Logs exported to S3

  ⏱️  Duration: 3.4s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rollback Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ✅ SUCCESS
Total Duration: 1m 52s
Downtime: 0s (instant traffic switch)

Rollback Details:
  Environment: production
  From Version: v1.3.0 (failed)
  To Version: v1.2.9 (stable)
  Strategy: Blue-Green Switch

Current State:
  Active Environment: green (v1.2.9)
  Replicas: 3/3 healthy
  Error Rate: 0.01% ✅
  Response Time: 84ms ✅
  Traffic: 100% on green

Root Causes:
  1. Memory leak in cache service (CRITICAL)
  2. N+1 database query (HIGH)
  3. Missing error handling (MEDIUM)

Next Steps:
  1. Review RCA report: /tmp/rca-v1.3.0-failure.md
  2. Create bug fix PR addressing root causes
  3. Add integration tests for identified issues
  4. Re-deploy v1.3.1 with fixes

Notifications Sent:
  ✅ Slack: #incidents (critical alert)
  ✅ PagerDuty: Incident created
  ✅ Email: engineering@example.com
  ✅ GitHub: Issue created (#1234)

✅ Rollback Complete! System Stable.
```

## Chains With

```bash
# Rollback → analyze → create bug ticket
/workflow:rollback --environment production && \
/agent-rca --analyze-logs true

# Rollback → notify → schedule fix
/workflow:rollback --environment staging --reason "Bug found" && \
/slack-notify "#engineering" "Rollback completed, fix needed"

# Immediate production rollback
/workflow:rollback --environment production --immediate true --reason "Critical outage"
```

## See Also
- `/workflow:deployment` - Deployment orchestration
- `/workflow:cicd` - Full CI/CD pipeline
- `/agent-rca` - Root cause analysis agent
- `/monitoring-configure` - Setup monitoring alerts
- `/agent-health-check` - Monitor agent health
- `/smoke-test` - Post-rollback validation
