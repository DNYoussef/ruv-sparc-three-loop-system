---
name: workflow:deployment
description: Enhanced production deployment workflow with blue-green deployment, canary releases, and automated rollback
category: workflows
version: 2.0.0
requires_mcp: true
---

# 🚀 Deployment Workflow - Production Release Pipeline

**Workflow**: Production Deployment | **Category**: Workflows | **Loop**: Loop 3 Deployment

Comprehensive deployment workflow with blue-green deployment, canary releases, health checks, and automated rollback capabilities.

## Deployment Strategies

### 1. Blue-Green Deployment
```bash
#!/bin/bash
# Blue-Green Deployment Strategy

CURRENT_ENV=$(cat .current-env)  # "blue" or "green"
NEW_ENV=$([[ "$CURRENT_ENV" == "blue" ]] && echo "green" || echo "blue")

echo "🚀 Blue-Green Deployment"
echo "Current: ${CURRENT_ENV}"
echo "Target: ${NEW_ENV}"

# Deploy to inactive environment
echo "Deploying to ${NEW_ENV}..."
./deploy.sh --env ${NEW_ENV}

# Health check
echo "Running health checks..."
HEALTH_URL="https://${NEW_ENV}.example.com/health"
for i in {1..10}; do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" ${HEALTH_URL})
  if [ "$STATUS" = "200" ]; then
    echo "✅ Health check passed"
    break
  fi
  echo "Waiting for ${NEW_ENV} environment..."
  sleep 5
done

# Smoke tests
echo "Running smoke tests..."
npm run test:smoke -- --env ${NEW_ENV}

# Switch traffic
echo "Switching traffic to ${NEW_ENV}..."
aws elb modify-listener --load-balancer-arn $LB_ARN \
  --listener-port 443 \
  --default-actions TargetGroupArn=$NEW_ENV_TG_ARN

echo "${NEW_ENV}" > .current-env

echo "✅ Deployment complete. Old environment (${CURRENT_ENV}) still running for rollback."
```

### 2. Canary Deployment
```bash
#!/bin/bash
# Canary Release Strategy

VERSION=$1
CANARY_PERCENTAGE=10

echo "🐤 Canary Deployment - v${VERSION}"

# Deploy canary version
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-canary
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: myapp
        version: ${VERSION}
        track: canary
    spec:
      containers:
      - name: app
        image: myapp:${VERSION}
EOF

# Route 10% traffic to canary
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  selector:
    app: myapp
  ports:
  - port: 80
    targetPort: 8080
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "${CANARY_PERCENTAGE}"
spec:
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: app-canary
            port:
              number: 80
EOF

# Monitor canary for 30 minutes
echo "Monitoring canary (${CANARY_PERCENTAGE}% traffic)..."
for i in {1..30}; do
  ERROR_RATE=$(kubectl logs -l track=canary --tail=1000 | grep ERROR | wc -l)
  TOTAL_REQUESTS=$(kubectl logs -l track=canary --tail=1000 | wc -l)

  if [ $TOTAL_REQUESTS -gt 0 ]; then
    ERROR_PERCENTAGE=$((ERROR_RATE * 100 / TOTAL_REQUESTS))

    if [ $ERROR_PERCENTAGE -gt 5 ]; then
      echo "❌ Error rate too high (${ERROR_PERCENTAGE}%). Rolling back canary."
      kubectl delete deployment app-canary
      kubectl delete ingress app-ingress
      exit 1
    fi
  fi

  echo "✅ Canary healthy. Error rate: ${ERROR_PERCENTAGE}%"
  sleep 60
done

# Promote canary to stable
echo "✅ Canary successful. Promoting to stable..."
kubectl set image deployment/app-stable app=myapp:${VERSION}
kubectl delete deployment app-canary

echo "✅ Deployment complete"
```

## Complete Deployment Workflow

```bash
#!/bin/bash
# Complete Production Deployment

set -e

VERSION=$1
ENVIRONMENT=${2:-production}

echo "🚀 PRODUCTION DEPLOYMENT"
echo "========================="
echo "Version: ${VERSION}"
echo "Environment: ${ENVIRONMENT}"
echo "========================="

# Phase 1: Pre-Deployment Checks
echo "Phase 1: Pre-deployment checks..."

# Verify tests passed
if [ ! -f ".test-status" ] || [ "$(cat .test-status)" != "passed" ]; then
  echo "❌ Tests have not passed. Run /workflow:testing first."
  exit 1
fi

# Verify security scan
npm audit --audit-level=high

# Phase 2: Build
echo "Phase 2: Building application..."
npm run build

# Verify build
if [ ! -d "dist" ]; then
  echo "❌ Build failed"
  exit 1
fi

# Phase 3: Create Release Tag
echo "Phase 3: Creating release tag..."
git tag -a "v${VERSION}" -m "Release v${VERSION}"
git push origin "v${VERSION}"

# Phase 4: Docker Build
echo "Phase 4: Building Docker image..."
docker build -t myapp:${VERSION} .
docker tag myapp:${VERSION} myapp:latest

# Phase 5: Push to Registry
echo "Phase 5: Pushing to container registry..."
docker push myapp:${VERSION}
docker push myapp:latest

# Phase 6: Update Kubernetes Manifests
echo "Phase 6: Updating Kubernetes manifests..."
cat > k8s/deployment.yaml <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: myapp
        version: ${VERSION}
    spec:
      containers:
      - name: app
        image: myapp:${VERSION}
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
EOF

# Phase 7: Deploy to Kubernetes
echo "Phase 7: Deploying to Kubernetes..."
kubectl apply -f k8s/

# Phase 8: Monitor Rollout
echo "Phase 8: Monitoring rollout..."
kubectl rollout status deployment/myapp --timeout=5m

# Phase 9: Health Checks
echo "Phase 9: Running health checks..."
sleep 10

for i in {1..10}; do
  HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://api.example.com/health)

  if [ "$HEALTH_STATUS" = "200" ]; then
    echo "✅ Health check passed"
    break
  fi

  if [ $i -eq 10 ]; then
    echo "❌ Health checks failed after 10 attempts"
    kubectl rollout undo deployment/myapp
    exit 1
  fi

  sleep 5
done

# Phase 10: Smoke Tests
echo "Phase 10: Running smoke tests..."
npm run test:smoke

# Phase 11: Performance Validation
echo "Phase 11: Performance validation..."
npx autocannon https://api.example.com/api/users \
  -c 50 -d 10 > deployment/performance.json

# Check p95 latency
P95_LATENCY=$(jq '.latency.p95' deployment/performance.json)
if (( $(echo "$P95_LATENCY > 500" | bc -l) )); then
  echo "⚠️  Warning: P95 latency (${P95_LATENCY}ms) exceeds 500ms threshold"
fi

# Phase 12: Enable Monitoring
echo "Phase 12: Enabling monitoring..."
npx claude-flow@alpha memory store \
  --key "deployment/${VERSION}/metrics" \
  --value "{
    \"version\": \"${VERSION}\",
    \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\",
    \"status\": \"deployed\",
    \"p95_latency\": ${P95_LATENCY}
  }"

# Phase 13: Notify Team
echo "Phase 13: Sending deployment notification..."
curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "🚀 Production Deployment Complete",
    "attachments": [
      {
        "color": "good",
        "fields": [
          {"title": "Version", "value": "'"${VERSION}"'", "short": true},
          {"title": "Environment", "value": "'"${ENVIRONMENT}"'", "short": true},
          {"title": "P95 Latency", "value": "'"${P95_LATENCY}"'ms", "short": true},
          {"title": "Status", "value": "✅ Success", "short": true}
        ]
      }
    ]
  }'

echo "========================="
echo "✅ DEPLOYMENT COMPLETE"
echo "========================="
echo "Version: ${VERSION}"
echo "Environment: ${ENVIRONMENT}"
echo "P95 Latency: ${P95_LATENCY}ms"
echo "========================="
```

## Automated Rollback

```typescript
/**
 * Automated Rollback System
 */

interface RollbackConfig {
  errorRateThreshold: number;      // 5% error rate triggers rollback
  latencyThreshold: number;        // 1000ms p95 latency triggers rollback
  monitoringDuration: number;      // Monitor for 15 minutes
  checkInterval: number;           // Check every 30 seconds
}

const config: RollbackConfig = {
  errorRateThreshold: 5,
  latencyThreshold: 1000,
  monitoringDuration: 15 * 60 * 1000,
  checkInterval: 30 * 1000
};

async function monitorDeployment(
  version: string,
  config: RollbackConfig
): Promise<boolean> {
  const startTime = Date.now();

  while (Date.now() - startTime < config.monitoringDuration) {
    const metrics = await fetchMetrics();

    // Check error rate
    if (metrics.errorRate > config.errorRateThreshold) {
      console.error(`❌ Error rate ${metrics.errorRate}% exceeds threshold`);
      await rollback(version);
      return false;
    }

    // Check latency
    if (metrics.p95Latency > config.latencyThreshold) {
      console.error(`❌ P95 latency ${metrics.p95Latency}ms exceeds threshold`);
      await rollback(version);
      return false;
    }

    console.log(`✅ Metrics healthy: Error rate ${metrics.errorRate}%, P95 ${metrics.p95Latency}ms`);

    await new Promise(resolve => setTimeout(resolve, config.checkInterval));
  }

  console.log('✅ Deployment monitoring complete. No issues detected.');
  return true;
}

async function rollback(version: string): Promise<void> {
  console.log(`🔄 Initiating rollback from v${version}...`);

  // Get previous version
  const previousVersion = await getPreviousVersion(version);

  // Rollback Kubernetes deployment
  await exec(`kubectl rollout undo deployment/myapp`);

  // Wait for rollback to complete
  await exec(`kubectl rollout status deployment/myapp --timeout=5m`);

  // Verify health
  const healthStatus = await checkHealth();
  if (healthStatus) {
    console.log(`✅ Rollback to v${previousVersion} successful`);

    // Notify team
    await sendAlert({
      severity: 'warning',
      message: `Automatic rollback from v${version} to v${previousVersion}`,
      reason: 'Performance degradation detected'
    });
  } else {
    console.error('❌ Rollback failed! Manual intervention required!');
    await sendAlert({
      severity: 'critical',
      message: `Rollback from v${version} failed`,
      reason: 'Health checks failing after rollback'
    });
  }
}
```

---

## Deployment Checklist

### Pre-Deployment
- [ ] All tests passing (unit, integration, e2e)
- [ ] Security scan completed
- [ ] Performance benchmarks meet SLA
- [ ] Database migrations prepared
- [ ] Feature flags configured
- [ ] Rollback plan documented

### Deployment
- [ ] Version tag created
- [ ] Docker image built and pushed
- [ ] Kubernetes manifests updated
- [ ] Rolling update initiated
- [ ] Health checks passing
- [ ] Smoke tests passing

### Post-Deployment
- [ ] Monitoring enabled
- [ ] Metrics within acceptable range
- [ ] Error rates normal (< 0.1%)
- [ ] Performance metrics normal
- [ ] Team notified
- [ ] Documentation updated

---

## Integration with Three-Loop System

**Loop 1**: Release planning
**Loop 2**: Implementation complete
**Loop 3**: Production deployment (this workflow)

**Agent Status**: Production-Ready
**Version**: 2.0.0
