---
name: workflow:deployment
category: workflow
version: 1.0.0
---

# /workflow:deployment

Orchestrate multi-stage deployment workflows with validation gates.

## Usage
```bash
/workflow:deployment [options]
```

## Parameters
- `--environment` - Target environment: dev|staging|production (required)
- `--strategy` - Deployment strategy: rolling|blue-green|canary|recreate (default: rolling)
- `--validation` - Enable validation gates (default: true)
- `--smoke-test` - Run smoke tests post-deployment (default: true)
- `--auto-rollback` - Rollback on failure (default: true)
- `--notify` - Send deployment notifications (default: true)
- `--approve-manual` - Require manual approval for production (default: true)
- `--canary-percentage` - Canary traffic percentage (default: 10)

## What It Does

**Multi-Stage Deployment Workflow**:
1. 🔍 **Pre-Deployment**: Validate readiness
2. 🎯 **Target Selection**: Choose deployment target
3. 🚦 **Validation Gate**: Quality checks
4. 🚀 **Deploy**: Execute deployment strategy
5. 🏥 **Health Check**: Monitor pod/instance health
6. 🧪 **Smoke Test**: Run basic functionality tests
7. 📊 **Metrics**: Collect performance metrics
8. ✅ **Approval Gate**: Manual/auto approval
9. 📢 **Notify**: Send deployment status

**Deployment Strategies**:
- **Rolling**: Gradual pod replacement (zero downtime)
- **Blue-Green**: Full environment switch with instant rollback
- **Canary**: Gradual traffic shift (10% → 50% → 100%)
- **Recreate**: Terminate all, then deploy new (downtime)

## Examples

```bash
# Deploy to staging with rolling update
/workflow:deployment --environment staging

# Production deployment with blue-green
/workflow:deployment --environment production --strategy blue-green

# Canary deployment with 20% traffic
/workflow:deployment --environment production --strategy canary --canary-percentage 20

# Deploy without smoke tests
/workflow:deployment --environment dev --smoke-test false

# Deploy with manual approval
/workflow:deployment --environment production --approve-manual true

# Recreate deployment (downtime acceptable)
/workflow:deployment --environment dev --strategy recreate
```

## Output

```
🚀 Deployment Workflow Started

Environment: production
Strategy: Blue-Green
Validation Gates: Enabled
Auto-Rollback: Enabled

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gate 1: Pre-Deployment Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ All tests passing (452/452)
  ✅ Security audit: No critical issues
  ✅ Code coverage: 94.3% (≥80%)
  ✅ Docker image built: myapp:v1.3.0
  ✅ Image scanned: 0 vulnerabilities
  ✅ Manifests validated

  🎯 Ready to deploy
  ⏱️  Duration: 4.2s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gate 2: Target Environment Check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Cluster: production-us-east-1
  Namespace: myapp-prod
  Current Version: v1.2.9 (green environment)
  Target Version: v1.3.0 (blue environment)

  ✅ Cluster connectivity: OK
  ✅ Namespace exists: myapp-prod
  ✅ Resources available: CPU, Memory, Storage
  ✅ Blue environment: Ready for deployment
  ✅ Green environment: Active (serving traffic)

  ⏱️  Duration: 2.1s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage: Deploy to Blue Environment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🌊 Deploying v1.3.0 to blue environment...

  ✅ ConfigMaps updated
  ✅ Secrets synchronized
  ✅ Deployment scaled to 3 replicas

  Pod Status:
    myapp-blue-abc123: Running ✅ (health: OK)
    myapp-blue-def456: Running ✅ (health: OK)
    myapp-blue-ghi789: Running ✅ (health: OK)

  ✅ All pods healthy (3/3)
  ⏱️  Duration: 38.4s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gate 3: Health Check Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🏥 Performing health checks...

  Liveness Probes:
    ✅ /health → 200 OK (all pods)

  Readiness Probes:
    ✅ /ready → 200 OK (all pods)

  Dependency Checks:
    ✅ Database: Connected
    ✅ Redis: Connected
    ✅ External API: Reachable

  ⏱️  Duration: 5.3s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gate 4: Smoke Test Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🧪 Running smoke tests against blue environment...

  Critical Paths:
    ✅ User login: 87ms → Success
    ✅ Fetch dashboard: 124ms → Success
    ✅ Create record: 156ms → Success
    ✅ Update record: 143ms → Success
    ✅ Delete record: 98ms → Success

  API Endpoints:
    ✅ GET /api/users: 200 OK
    ✅ POST /api/auth/login: 200 OK
    ✅ GET /api/health: 200 OK
    ✅ GET /api/metrics: 200 OK

  ✅ All smoke tests passed (15/15)
  ⏱️  Duration: 8.7s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gate 5: Performance Metrics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📊 Collecting baseline metrics...

  Blue Environment (v1.3.0):
    Response Time: 87ms (target: <100ms) ✅
    Error Rate: 0.00% (target: <0.1%) ✅
    Throughput: 450 req/s ✅
    CPU: 28% ✅
    Memory: 32% ✅

  Green Environment (v1.2.9 - current):
    Response Time: 84ms
    Error Rate: 0.02%
    Throughput: 445 req/s
    CPU: 30%
    Memory: 35%

  Comparison:
    ⚡ Response Time: +3ms (+3.6%) - Acceptable
    ✅ Error Rate: -0.02% (improved)
    ✅ Throughput: +5 req/s (improved)
    ✅ CPU: -2% (improved)
    ✅ Memory: -3% (improved)

  🎯 Performance metrics within SLA
  ⏱️  Duration: 12.4s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gate 6: Manual Approval Required
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⏸️  Deployment paused for approval

  Summary:
    ✅ All validation gates passed
    ✅ Blue environment ready
    ✅ Performance within SLA
    ✅ No regressions detected

  Ready to switch traffic from green → blue

  [APPROVED] (auto-approved after 30s timeout)
  ⏱️  Duration: 2.1s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage: Traffic Switch (Blue-Green)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔀 Switching traffic from green to blue...

  ✅ Service updated: myapp-svc → blue selector
  ✅ Ingress updated: myapp.example.com → blue backend
  ✅ Load balancer health check: 3/3 targets healthy
  ✅ DNS propagated

  Traffic Routing:
    Blue (v1.3.0): 100% ✅
    Green (v1.2.9): 0%

  ⏱️  Duration: 8.2s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage: Post-Switch Monitoring (5min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📊 Monitoring blue environment...

  [00:30] Error Rate: 0.00% ✅ | Response Time: 88ms ✅
  [01:00] Error Rate: 0.00% ✅ | Response Time: 86ms ✅
  [01:30] Error Rate: 0.01% ✅ | Response Time: 89ms ✅
  [02:00] Error Rate: 0.00% ✅ | Response Time: 85ms ✅
  [02:30] Error Rate: 0.00% ✅ | Response Time: 87ms ✅

  ✅ No anomalies detected
  ✅ Traffic stable
  ⏱️  Duration: 5m 0s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage: Green Environment Cleanup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🧹 Scaling down green environment...

  ✅ Green deployment scaled to 0
  ✅ Resources released
  ✅ Old image retained (rollback capability)

  ⏱️  Duration: 4.3s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Deployment Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ✅ SUCCESS
Strategy: Blue-Green
Total Duration: 7m 23s
Downtime: 0s (zero-downtime deployment)

Deployment Details:
  Environment: production
  Version: v1.2.9 → v1.3.0
  Replicas: 3/3 healthy
  Traffic: 100% on blue

Performance:
  Response Time: 87ms (SLA: <100ms) ✅
  Error Rate: 0.00% (SLA: <0.1%) ✅
  Throughput: 450 req/s ✅

URLs:
  🌐 Production: https://myapp.example.com
  📊 Metrics: https://grafana.example.com/d/myapp
  📊 Logs: kubectl logs -n myapp-prod -l app=myapp,env=blue

Rollback Command (if needed):
  /workflow:rollback --environment production --to v1.2.9

Notifications Sent:
  ✅ Slack: #deployments
  ✅ Email: devops@example.com
  ✅ PagerDuty: Deployment success event

✅ Deployment Complete!
```

## Chains With

```bash
# Full deployment pipeline
/docker-build && /workflow:deployment --environment staging

# Deploy to staging then production
/workflow:deployment --environment staging && \
/smoke-test && \
/workflow:deployment --environment production

# Deploy with monitoring
/workflow:deployment --environment production && \
/monitoring-configure --alerts true
```

## See Also
- `/workflow:cicd` - Full CI/CD pipeline
- `/workflow:rollback` - Rollback deployment
- `/docker-build` - Build container images
- `/k8s-deploy` - Kubernetes deployment
- `/smoke-test` - Post-deployment testing
- `/monitoring-configure` - Setup monitoring
