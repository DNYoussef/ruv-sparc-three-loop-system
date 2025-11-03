# Test 2: Kubernetes Cluster Deployment and Management

## Test Objective
Validate Kubernetes deployment automation using kubectl, Helm, and deployment scripts across local (Minikube/Kind) and cloud (GKE/EKS) environments.

## Prerequisites
- kubectl >= 1.27 installed
- Helm >= 3.12 installed
- Docker installed
- Minikube or Kind for local testing
- Access to GKE or EKS for cloud testing
- `jq` utility installed

## Test Scenarios

### Scenario 2.1: Local Kubernetes Cluster Setup

**Setup:**
```bash
# Create Kind cluster
cat > kind-config.yaml << 'EOF'
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraPortMappings:
  - containerPort: 30080
    hostPort: 8080
  - containerPort: 30443
    hostPort: 8443
- role: worker
- role: worker
EOF

kind create cluster --name test-cluster --config kind-config.yaml

# Verify cluster
kubectl cluster-info
kubectl get nodes
```

**Test Execution:**
```bash
# Deploy test application using kubectl
bash ../resources/scripts/deploy_k8s.sh \
  kubectl \
  ../resources/templates/k8s-deployment.yaml \
  production

# Check deployment status
bash ../resources/scripts/deploy_k8s.sh \
  status \
  myapp \
  production

# Scale deployment
bash ../resources/scripts/deploy_k8s.sh \
  scale \
  myapp \
  5 \
  production

# Verify scaling
kubectl get deployment myapp -n production
kubectl get pods -n production -l app=myapp
```

**Expected Results:**
- Kind cluster created with 3 nodes
- Deployment created in production namespace
- All pods running and ready
- Scaling to 5 replicas successful
- No pending or failed pods

**Success Criteria:**
- ✅ Cluster has 3 nodes (1 control-plane, 2 workers)
- ✅ All nodes in Ready state
- ✅ Deployment successfully created
- ✅ 5/5 pods running after scaling
- ✅ All health checks passing

---

### Scenario 2.2: Helm Chart Deployment

**Setup:**
```bash
# Create Helm chart
helm create test-chart
cd test-chart

# Customize values
cat > values.yaml << 'EOF'
replicaCount: 3

image:
  repository: nginx
  tag: alpine
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-staging"
  hosts:
    - host: test.local
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: test-tls
      hosts:
        - test.local

resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 200m
    memory: 256Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
EOF
```

**Test Execution:**
```bash
# Deploy Helm chart
bash ../resources/scripts/deploy_k8s.sh \
  helm \
  test-release \
  ./test-chart \
  production \
  values.yaml

# Verify Helm release
helm list -n production
helm status test-release -n production

# Get deployment details
kubectl get all -n production -l app.kubernetes.io/instance=test-release

# Test Helm upgrade
echo "replicaCount: 5" >> values.yaml
helm upgrade test-release ./test-chart -n production -f values.yaml

# Verify upgrade
kubectl get deployment -n production -l app.kubernetes.io/instance=test-release
```

**Expected Results:**
- Helm chart deployed successfully
- 3 replicas initially created
- Service and Ingress created
- HPA configured correctly
- Upgrade to 5 replicas successful
- Release history tracked

**Success Criteria:**
- ✅ `helm list` shows test-release as deployed
- ✅ All pods running (3 initially, 5 after upgrade)
- ✅ Service accessible within cluster
- ✅ HPA created with correct settings
- ✅ Ingress resource created

---

### Scenario 2.3: ConfigMap and Secret Management

**Setup:**
```bash
# Create test configuration file
cat > app-config.properties << 'EOF'
database.host=postgres.production.svc.cluster.local
database.port=5432
cache.ttl=300
log.level=info
EOF

# Create test secrets file
cat > secrets.env << 'EOF'
DATABASE_PASSWORD=supersecret123
API_KEY=sk_test_1234567890
JWT_SECRET=jwt_secret_key_456
EOF
```

**Test Execution:**
```bash
# Create ConfigMap
bash ../resources/scripts/deploy_k8s.sh \
  configmap \
  app-config \
  app-config.properties \
  production

# Verify ConfigMap
kubectl get configmap app-config -n production -o yaml

# Create Secret
bash ../resources/scripts/deploy_k8s.sh \
  secret \
  app-secrets \
  secrets.env \
  production

# Verify Secret (without exposing values)
kubectl get secret app-secrets -n production
kubectl describe secret app-secrets -n production

# Deploy pod using ConfigMap and Secret
cat > test-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: config-test
  namespace: production
spec:
  containers:
  - name: test
    image: busybox
    command: ["sh", "-c", "env && sleep 3600"]
    envFrom:
    - configMapRef:
        name: app-config
    - secretRef:
        name: app-secrets
EOF

kubectl apply -f test-pod.yaml

# Verify environment variables
kubectl logs config-test -n production | grep -E "database|API_KEY"
```

**Expected Results:**
- ConfigMap created with config file content
- Secret created with encrypted data
- Pod successfully mounts ConfigMap and Secret
- Environment variables accessible in pod
- Secrets not visible in plain text

**Success Criteria:**
- ✅ ConfigMap created with correct data
- ✅ Secret created and base64 encoded
- ✅ Pod runs successfully with configs
- ✅ Environment variables populated
- ✅ No plain text secrets in kubectl describe

---

### Scenario 2.4: Deployment Strategies (Rolling Update and Rollback)

**Setup:**
```bash
# Deploy initial version
cat > deployment-v1.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rolling-demo
  namespace: production
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: rolling-demo
  template:
    metadata:
      labels:
        app: rolling-demo
        version: v1
    spec:
      containers:
      - name: app
        image: nginx:1.24-alpine
        ports:
        - containerPort: 80
EOF

kubectl apply -f deployment-v1.yaml
kubectl wait --for=condition=available deployment/rolling-demo -n production --timeout=60s
```

**Test Execution:**
```bash
# Update to v2
cat > deployment-v2.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rolling-demo
  namespace: production
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: rolling-demo
  template:
    metadata:
      labels:
        app: rolling-demo
        version: v2
    spec:
      containers:
      - name: app
        image: nginx:1.25-alpine
        ports:
        - containerPort: 80
EOF

# Apply update
kubectl apply -f deployment-v2.yaml

# Watch rollout
kubectl rollout status deployment/rolling-demo -n production

# Check revision history
kubectl rollout history deployment/rolling-demo -n production

# Introduce bad update (v3 with wrong image)
kubectl set image deployment/rolling-demo app=nginx:broken-tag -n production

# Watch rollout (should fail)
kubectl rollout status deployment/rolling-demo -n production --timeout=120s || echo "Rollout failed as expected"

# Rollback to previous version
bash ../resources/scripts/deploy_k8s.sh \
  rollback \
  rolling-demo \
  production

# Verify rollback
kubectl rollout status deployment/rolling-demo -n production
kubectl get pods -n production -l app=rolling-demo -o jsonpath='{.items[0].spec.containers[0].image}'
```

**Expected Results:**
- v1 deploys successfully with 5 replicas
- Rolling update to v2 completes with zero downtime
- v3 rollout fails due to bad image
- Rollback restores v2 successfully
- All pods running after rollback
- Revision history tracked

**Success Criteria:**
- ✅ Initial deployment successful (5/5 pods)
- ✅ Rolling update completes without downtime
- ✅ Failed rollout stops automatically
- ✅ Rollback completes successfully
- ✅ Correct version restored after rollback

---

### Scenario 2.5: GKE/EKS Cluster Deployment

**GKE Test:**
```bash
# Deploy to GKE
bash ../resources/scripts/gcp_deploy.sh \
  create-gke \
  test-gke-cluster \
  us-central1-a \
  e2-medium \
  3 \
  true

# Wait for cluster to be ready
gcloud container clusters describe test-gke-cluster \
  --zone=us-central1-a \
  --format="value(status)"

# Deploy application
bash ../resources/scripts/gcp_deploy.sh \
  gke \
  test-gke-cluster \
  us-central1-a \
  ../resources/templates/k8s-deployment.yaml \
  production

# Verify deployment
kubectl get all -n production

# Test autoscaling
kubectl run -i --tty load-generator --rm --image=busybox --restart=Never -n production -- /bin/sh
# Inside pod: while true; do wget -q -O- http://myapp.production.svc.cluster.local; done
```

**Expected Results (GKE):**
- GKE cluster created with 3 nodes
- Cluster autoscaling enabled
- Application deployed successfully
- HPA scaling based on CPU load
- Cluster handles load gracefully

**Success Criteria (GKE):**
- ✅ Cluster status is RUNNING
- ✅ All 3 nodes in Ready state
- ✅ Application pods running
- ✅ Autoscaling triggers under load
- ✅ Cluster scales nodes when needed

---

## Performance Tests

### Load Testing
```bash
# Deploy load testing tool
kubectl create deployment load-test \
  --image=williamyeh/hey \
  --namespace=production

# Run load test
kubectl run load-test -i --tty --rm \
  --image=williamyeh/hey \
  --namespace=production \
  --restart=Never \
  -- -z 60s -c 50 http://myapp.production.svc.cluster.local

# Monitor HPA
watch kubectl get hpa -n production
```

### Resource Monitoring
```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n production

# Check events
kubectl get events -n production --sort-by='.lastTimestamp'

# Check pod logs
kubectl logs -f deployment/myapp -n production --tail=50
```

---

## Cleanup

```bash
# Delete local Kind cluster
kind delete cluster --name test-cluster

# Delete GKE cluster
gcloud container clusters delete test-gke-cluster \
  --zone=us-central1-a \
  --quiet

# Delete namespace (if using existing cluster)
kubectl delete namespace production --grace-period=30
```

## Test Report Template

```markdown
# Kubernetes Deployment Test Report

**Date:** YYYY-MM-DD
**Tester:** Name
**Environment:** Kind/GKE/EKS

## Test Results Summary

| Scenario | Status | Duration | Pods Running | Notes |
|----------|--------|----------|--------------|-------|
| 2.1 Local Cluster | ✅/❌ | XXs | X/X | |
| 2.2 Helm Chart | ✅/❌ | XXs | X/X | |
| 2.3 ConfigMap/Secret | ✅/❌ | XXs | X/X | |
| 2.4 Rolling Update | ✅/❌ | XXs | X/X | |
| 2.5 GKE/EKS | ✅/❌ | XXXs | X/X | |

## Performance Metrics

- **Deployment Time:** XXs
- **Pod Startup Time:** XXs
- **Scaling Time:** XXs (X to Y replicas)
- **Rollback Time:** XXs
- **Load Test RPS:** XXX requests/sec
- **P95 Latency:** XXms

## Issues Encountered

1. **Issue:** Description
   - **Resolution:** Fix applied
   - **Impact:** Severity

## Recommendations

- Best practices observed
- Script improvements
- Documentation updates needed

## Sign-off

- [ ] All tests passed
- [ ] Cleanup completed
- [ ] Documentation updated
```

## Notes

- Test in non-production environment first
- Monitor resource usage during tests
- Clean up resources to avoid charges
- Keep kubectl and Helm versions updated
- Use namespaces to isolate test workloads
- Verify RBAC permissions before deploying
