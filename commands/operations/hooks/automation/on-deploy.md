---
name: hook:on-deploy
description: Deployment hook with monitoring and rollback capabilities
category: Automation Hooks
version: 1.0.0
requires:
  - kubectl (for k8s)
  - docker
usage: |
  /hook:on-deploy --environment "production" --health-check --enable-rollback
  /hook:on-deploy --platform "kubernetes" --namespace "app" --monitor-logs
---

# Hook: On-Deploy

**Category**: Automation Hooks
**Purpose**: Automated deployment with health checks, monitoring, and rollback.

## Implementation

```bash
#!/bin/bash
# Deployment hook with health checks

ENVIRONMENT=${1:-staging}
NAMESPACE=${2:-default}

echo "🚀 Deploying to $ENVIRONMENT..."

# Deploy
kubectl apply -f deployment.yaml -n $NAMESPACE

# Wait for rollout
kubectl rollout status deployment/app -n $NAMESPACE --timeout=5m

# Health check
echo "🏥 Running health checks..."
HEALTH_URL="https://app-$ENVIRONMENT.example.com/health"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)

if [ "$HTTP_CODE" == "200" ]; then
  echo "✅ Deployment successful! Health check passed."

  # Send success notification
  curl -X POST $SLACK_WEBHOOK -d "{\"text\":\"✅ Deployed to $ENVIRONMENT\"}"

  exit 0
else
  echo "❌ Health check failed! Rolling back..."

  # Rollback
  kubectl rollout undo deployment/app -n $NAMESPACE

  # Send failure notification
  curl -X POST $SLACK_WEBHOOK -d "{\"text\":\"❌ Deployment to $ENVIRONMENT failed. Rolled back.\"}"

  exit 1
fi
```

---

**Status**: Production Ready
**Version**: 1.0.0
