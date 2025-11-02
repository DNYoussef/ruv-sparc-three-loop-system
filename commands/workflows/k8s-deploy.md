---
name: k8s-deploy
category: deployment
version: 1.0.0
---

# /k8s-deploy

Deploy applications to Kubernetes clusters with intelligent orchestration.

## Usage
```bash
/k8s-deploy [manifest_path] [options]
```

## Parameters
- `manifest_path` - Path to K8s manifests (default: k8s/)
- `--namespace` - Target namespace (default: default)
- `--context` - Kubectl context (default: current)
- `--cluster` - Target cluster name (required for multi-cluster)
- `--environment` - Environment: dev|staging|production (default: staging)
- `--replicas` - Override replica count (optional)
- `--strategy` - Deployment strategy: rolling|recreate|blue-green|canary (default: rolling)
- `--dry-run` - Preview changes without applying (default: false)
- `--auto-scale` - Enable HPA (Horizontal Pod Autoscaler) (default: false)
- `--wait-ready` - Wait for pods to be ready (default: true)
- `--timeout` - Deployment timeout in seconds (default: 300)

## What It Does

**Comprehensive Kubernetes Deployment**:
1. 🔍 Validate manifests (kubeval, kube-linter)
2. 🎯 Select target cluster and namespace
3. 🏗️ Apply ConfigMaps and Secrets
4. 📦 Deploy Deployments/StatefulSets
5. 🌐 Configure Services and Ingress
6. 📊 Monitor rollout progress
7. 🏥 Health check validation
8. 🔄 Auto-rollback on failure
9. 📈 Post-deployment metrics

**Deployment Strategies**:
- **Rolling Update**: Zero-downtime gradual rollout
- **Recreate**: Terminate old, then create new
- **Blue-Green**: Full environment switch
- **Canary**: Gradual traffic shift (10%→50%→100%)

## Examples

```bash
# Basic deployment
/k8s-deploy k8s/manifests/

# Deploy to specific namespace and cluster
/k8s-deploy --namespace myapp-prod --cluster production-us-east

# Dry run to preview changes
/k8s-deploy --dry-run true

# Canary deployment
/k8s-deploy --strategy canary --environment production

# Deploy with auto-scaling
/k8s-deploy --auto-scale true --replicas 3

# Blue-green deployment
/k8s-deploy --strategy blue-green --namespace production

# Deploy specific manifest
/k8s-deploy k8s/deployment.yaml --namespace staging --replicas 5
```

## Output

```
☸️  Kubernetes Deployment Started

Cluster: production-us-east-1
Namespace: myapp-prod
Context: arn:aws:eks:us-east-1:123456789:cluster/prod
Strategy: Rolling Update

Validating Manifests:
  ✅ deployment.yaml: Valid
  ✅ service.yaml: Valid
  ✅ ingress.yaml: Valid
  ✅ configmap.yaml: Valid
  ⚠️  No resource limits defined (consider adding)
  ✅ kubeval: All manifests valid
  ✅ kube-linter: No critical issues

Pre-Deployment Checks:
  ✅ Cluster connectivity: OK
  ✅ Namespace exists: myapp-prod
  ✅ RBAC permissions: Verified
  ✅ Image pull secrets: Configured
  ✅ Storage class available: gp3
  ✅ Current pods: 3/3 healthy

Applying Resources:
  📝 ConfigMap/myapp-config: Created
  🔐 Secret/myapp-secrets: Created (from vault)
  📦 Deployment/myapp-web: Updated
  🌐 Service/myapp-svc: Unchanged
  🌍 Ingress/myapp-ingress: Updated (added new route)

Rolling Update Progress:
  🔄 Scaling new ReplicaSet to 1...
     Pod myapp-web-v2-abc123: Scheduled → Pending → Running
     Health Check: ✅ Passed (2.3s)

  🔄 Scaling new ReplicaSet to 2...
     Pod myapp-web-v2-def456: Scheduled → Pending → Running
     Health Check: ✅ Passed (2.1s)

  🔄 Scaling down old ReplicaSet to 2...
     Pod myapp-web-v1-xyz789: Terminating → Terminated

  🔄 Scaling new ReplicaSet to 3...
     Pod myapp-web-v2-ghi789: Scheduled → Pending → Running
     Health Check: ✅ Passed (2.4s)

  🔄 Scaling down old ReplicaSet to 0...
     Pod myapp-web-v1-aaa111: Terminating → Terminated
     Pod myapp-web-v1-bbb222: Terminating → Terminated

Deployment Status:
  ✅ Desired: 3 replicas
  ✅ Current: 3 replicas
  ✅ Updated: 3 replicas
  ✅ Available: 3 replicas
  ✅ Ready: 3/3 pods

Post-Deployment Validation:
  ✅ All pods healthy
  ✅ Service endpoints updated
  ✅ Ingress routing: myapp.example.com → 200 OK
  ✅ Load balancer healthy
  ✅ Auto-scaling configured (min: 3, max: 10, CPU: 70%)

Resource Usage:
  CPU: 450m / 1500m (30%)
  Memory: 1.2Gi / 4Gi (30%)
  Storage: 5Gi / 20Gi (25%)

Deployment Metrics:
  Total Time: 42.6s
  Downtime: 0s (zero-downtime)
  Rollout Duration: 35.2s
  Average Pod Startup: 2.3s

Deployed Resources:
  🌐 External URL: https://myapp.example.com
  🌐 Internal Service: myapp-svc.myapp-prod.svc.cluster.local:8080
  📊 Metrics: https://grafana.example.com/d/k8s-myapp
  📊 Logs: kubectl logs -n myapp-prod -l app=myapp-web

Rollback Command (if needed):
  kubectl rollout undo deployment/myapp-web -n myapp-prod
  OR: /workflow:rollback --namespace myapp-prod --deployment myapp-web

✅ Kubernetes Deployment Complete!
```

## Chains With

```bash
# Build → deploy pipeline
/docker-build --tag myapp:v1.0.0 && /k8s-deploy

# Deploy → smoke test → monitor
/k8s-deploy --environment production && /smoke-test && /monitoring-configure

# Full GitOps workflow
/github-release && /docker-build && /k8s-deploy --strategy canary

# Multi-environment deployment
/k8s-deploy --environment staging && \
/smoke-test && \
/k8s-deploy --environment production --strategy blue-green
```

## See Also
- `/docker-build` - Build container images
- `/docker-deploy` - Docker deployment
- `/workflow:deployment` - Deployment workflow
- `/workflow:rollback` - Rollback deployment
- `/monitoring-configure` - Setup K8s monitoring
- `/smoke-test` - Post-deployment validation
- `/load-test` - Load testing for scaled deployments
