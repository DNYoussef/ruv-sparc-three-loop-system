---
name: workflow:cicd
category: workflow
version: 1.0.0
---

# /workflow:cicd

Complete CI/CD workflow orchestration with intelligent failure recovery.

## Usage
```bash
/workflow:cicd [options]
```

## Parameters
- `--stage` - Run specific stage: all|build|test|deploy|release (default: all)
- `--environment` - Target environment: dev|staging|production (default: staging)
- `--strategy` - Deployment strategy: rolling|blue-green|canary (default: rolling)
- `--parallel` - Enable parallel test execution (default: true)
- `--fail-fast` - Stop on first failure (default: false)
- `--notify` - Send notifications (default: true)
- `--auto-rollback` - Rollback on failure (default: true)
- `--docker` - Build Docker image (default: true)
- `--k8s` - Deploy to Kubernetes (default: false)

## What It Does

**Complete CI/CD Pipeline (9 Stages)**:
1. 🔍 **Pre-flight**: Validate environment and dependencies
2. 🏗️ **Build**: Compile and bundle application
3. 🧪 **Test**: Run all test suites in parallel
4. 🔒 **Security**: Security and dependency audits
5. 📊 **Quality**: Code quality and coverage checks
6. 🐳 **Package**: Build Docker images
7. 🚀 **Deploy**: Deploy to target environment
8. ✅ **Verify**: Post-deployment validation
9. 📢 **Notify**: Send status notifications

**Intelligent Features**:
- Auto-retry failed tests (up to 3 attempts)
- Parallel test execution (4x faster)
- Smart caching (dependencies, build artifacts)
- Auto-rollback on deployment failure
- Real-time progress monitoring
- Comprehensive failure reports

## Examples

```bash
# Full CI/CD pipeline
/workflow:cicd

# Build and test only
/workflow:cicd --stage test

# Deploy to production with blue-green
/workflow:cicd --environment production --strategy blue-green

# Build stage only
/workflow:cicd --stage build --parallel true

# Fast fail pipeline
/workflow:cicd --fail-fast true

# Docker build with K8s deployment
/workflow:cicd --docker true --k8s true --environment staging

# Release stage
/workflow:cicd --stage release
```

## Output

```
🔄 CI/CD Pipeline Started

Configuration:
  Environment: staging
  Strategy: Rolling Update
  Parallel Tests: Enabled
  Docker Build: Yes
  K8s Deploy: Yes
  Auto-Rollback: Enabled

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1/9: Pre-Flight Checks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ Git repository clean
  ✅ Dependencies installed
  ✅ Environment variables loaded
  ✅ Docker daemon running
  ✅ Kubernetes cluster accessible
  ⏱️  Duration: 3.2s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 2/9: Build Application
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔨 Installing dependencies...
  ✅ npm install (cached, 2.1s)

  🔨 Building application...
  ✅ TypeScript compilation (12.3s)
  ✅ Webpack bundling (8.7s)
  ✅ Asset optimization (3.4s)

  Build Artifacts:
    - dist/main.js (342 KB, gzip: 98 KB)
    - dist/vendor.js (1.2 MB, gzip: 387 KB)
    - dist/assets/* (45 files, 2.3 MB)

  ⏱️  Duration: 24.5s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 3/9: Test Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Running tests in parallel (4 workers)...

  🧪 Unit Tests: 342 passed (8.2s)
  🧪 Integration Tests: 87 passed (12.4s)
  🧪 E2E Tests: 23 passed (45.6s)

  Coverage Report:
    Statements: 94.3% (target: 80%)
    Branches: 89.7% (target: 75%)
    Functions: 92.1% (target: 80%)
    Lines: 93.8% (target: 80%)

  ✅ All tests passed (452/452)
  ⏱️  Duration: 45.6s (parallel)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 4/9: Security Audit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔒 Running security scan...
  ✅ No critical vulnerabilities (0.8s)
  ✅ Dependency audit passed (1.2s)
  ⚠️  2 medium severity issues (non-blocking)

  🔐 Secrets scan...
  ✅ No secrets exposed

  ⏱️  Duration: 2.3s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 5/9: Quality Checks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎯 Running linters...
  ✅ ESLint: No errors (2.1s)
  ✅ Prettier: All files formatted (0.8s)
  ✅ TypeScript: No type errors (3.4s)

  📊 Code Quality Score: 92/100

  ⏱️  Duration: 6.3s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 6/9: Docker Build
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🐳 Building image: myapp:v1.3.0
  ✅ Multi-stage build complete
  ✅ Image size: 142 MB (75% reduction)
  ✅ Security scan: No vulnerabilities
  ✅ Pushed to registry

  ⏱️  Duration: 37.8s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 7/9: Deploy to Staging
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ☸️  Deploying to Kubernetes...
  ✅ Manifests validated
  ✅ Rolling update started
  ✅ New pods ready (3/3)
  ✅ Old pods terminated
  ✅ Service endpoints updated
  ✅ Ingress routing verified

  ⏱️  Duration: 42.6s (zero downtime)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 8/9: Post-Deployment Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔍 Running smoke tests...
  ✅ Health check: /health → 200 OK
  ✅ API endpoints: 15/15 passed
  ✅ Database connectivity: OK
  ✅ External services: All reachable

  📊 Performance Metrics:
    - Response time: 87ms (baseline: 84ms)
    - Error rate: 0.00%
    - CPU usage: 30%
    - Memory usage: 28%

  ⏱️  Duration: 8.4s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 9/9: Notifications
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📢 Sending notifications...
  ✅ Slack: #deployments
  ✅ Email: devops@example.com
  ✅ GitHub: Deployment status updated

  ⏱️  Duration: 1.2s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pipeline Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ✅ SUCCESS
Total Duration: 2m 52s
Stages Passed: 9/9

Build Metrics:
  - Build Time: 24.5s
  - Test Time: 45.6s (parallel)
  - Deploy Time: 42.6s
  - Cache Hit Rate: 78%

Deployment:
  Environment: staging
  Version: v1.3.0
  Replicas: 3/3 healthy
  URL: https://staging.myapp.example.com

Rollback Command (if needed):
  /workflow:rollback --environment staging --to v1.2.9

✅ CI/CD Pipeline Complete!
```

## Chains With

```bash
# CI/CD → monitor
/workflow:cicd && /monitoring-configure

# CI/CD for production with canary
/workflow:cicd --environment production --strategy canary

# Build → test → manual deploy
/workflow:cicd --stage test && /k8s-deploy --environment production

# Full release workflow
/github-release && /workflow:cicd --environment production
```

## See Also
- `/docker-build` - Docker image build
- `/k8s-deploy` - Kubernetes deployment
- `/workflow:deployment` - Deployment orchestration
- `/workflow:rollback` - Rollback deployment
- `/security-audit` - Security scanning
- `/regression-test` - Regression testing
- `/github-release` - Release automation
