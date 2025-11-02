---
name: docker-deploy
category: deployment
version: 1.0.0
---

# /docker-deploy

Deploy Docker containers to registries and orchestration platforms.

## Usage
```bash
/docker-deploy [image_tag] [options]
```

## Parameters
- `image_tag` - Docker image tag to deploy (required)
- `--target` - Deployment target: registry|docker-compose|swarm|ecs (default: registry)
- `--registry` - Registry URL (default: docker.io)
- `--environment` - Target environment: dev|staging|production (default: staging)
- `--replicas` - Number of replicas (default: 1)
- `--health-check` - Enable health check validation (default: true)
- `--rollback-on-failure` - Auto-rollback on deployment failure (default: true)
- `--blue-green` - Use blue-green deployment strategy (default: false)

## What It Does

**Intelligent Deployment Process**:
1. 🔍 Pre-deployment validation
2. 🚀 Push image to registry
3. 🎯 Deploy to target platform
4. 🏥 Health check monitoring
5. 📊 Rollout progress tracking
6. 🔄 Auto-rollback on failure
7. 📈 Post-deployment metrics
8. 🔔 Deployment notifications

**Deployment Targets**:
- **Registry**: Docker Hub, GitHub Container Registry, ECR, GCR
- **Docker Compose**: Local/remote compose deployments
- **Docker Swarm**: Swarm service updates
- **Amazon ECS**: Fargate/EC2 task definitions
- **Kubernetes**: Via kubectl integration

## Examples

```bash
# Deploy to registry
/docker-deploy myapp:v1.2.3

# Deploy to production with health checks
/docker-deploy myapp:v1.2.3 --environment production --health-check true

# Deploy to Docker Swarm with replicas
/docker-deploy myapp:latest --target swarm --replicas 3

# Blue-green deployment
/docker-deploy myapp:v2.0.0 --blue-green true --environment production

# Deploy to ECS
/docker-deploy myapp:prod --target ecs --replicas 2

# Deploy with custom registry
/docker-deploy myapp:v1.0.0 --registry ghcr.io/myorg --environment staging
```

## Output

```
🚀 Docker Deployment Started

Image: myapp:v1.2.3
Target: Docker Hub (docker.io)
Environment: production
Strategy: Rolling Update

Pre-Deployment Checks:
  ✅ Image exists locally
  ✅ Registry authentication successful
  ✅ Image signature verified
  ✅ Security scan passed (0 critical issues)
  ✅ Size: 142 MB (within limits)

Pushing to Registry:
  📤 Layer 1/8: Pushed [====================] 100%
  📤 Layer 2/8: Pushed [====================] 100%
  ...
  ✅ Push complete: docker.io/myorg/myapp:v1.2.3

Deployment Configuration:
  Platform: Docker Swarm
  Replicas: 3
  Update Strategy: Rolling (max 1 unavailable)
  Health Check: HTTP GET /health (30s timeout)
  Rollback: Auto (on health check failure)

Deploying Services:
  🔄 Creating service: myapp-web
  🔄 Scaling to 3 replicas...

  Replica 1/3: Starting... ⏳
              Health Check... ✅ Healthy (1.2s)
  Replica 2/3: Starting... ⏳
              Health Check... ✅ Healthy (1.4s)
  Replica 3/3: Starting... ⏳
              Health Check... ✅ Healthy (1.1s)

Post-Deployment Validation:
  ✅ All replicas healthy (3/3)
  ✅ Load balancer updated
  ✅ Old version scaled down
  ✅ DNS records updated

Deployment Metrics:
  Total Time: 45.2s
  Downtime: 0s (zero-downtime deployment)
  Success Rate: 100%
  Average Response Time: 87ms (+3ms from baseline)

Deployed Endpoints:
  🌐 Production: https://myapp.production.example.com
  🌐 Internal: http://myapp.svc.cluster.local:8080
  📊 Metrics: https://grafana.example.com/d/myapp

Rollback Command (if needed):
  /docker-deploy myapp:v1.2.2 --environment production --replicas 3

✅ Deployment Complete!
```

## Chains With

```bash
# Build → deploy pipeline
/docker-build --tag myapp:v1.0.0 && /docker-deploy myapp:v1.0.0

# Deploy → smoke test → monitor
/docker-deploy myapp:latest && /smoke-test && /monitoring-configure

# Full CI/CD
/security-audit && /docker-build && /docker-deploy --environment production

# Deploy with rollback on failure
/docker-deploy myapp:v2.0.0 || /workflow:rollback --to v1.9.0
```

## See Also
- `/docker-build` - Build Docker images
- `/k8s-deploy` - Kubernetes deployment
- `/workflow:deployment` - Deployment workflow orchestration
- `/workflow:rollback` - Rollback deployment
- `/monitoring-configure` - Setup deployment monitoring
- `/smoke-test` - Post-deployment validation
