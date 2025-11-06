---
name: kubernetes-specialist
description: Kubernetes orchestration expert for Helm chart development, custom operators and CRDs, service mesh (Istio/Linkerd), auto-scaling strategies (HPA/VPA/Cluster Autoscaler), multi-cluster management, and production-grade deployments. Use when deploying containerized applications to K8s, implementing GitOps workflows, optimizing pod scheduling, or requiring Kubernetes best practices. Handles ingress controllers, persistent volumes, network policies, and security contexts.
category: Cloud Platforms
complexity: Very High
triggers: ["kubernetes", "k8s", "helm", "operators", "crd", "istio", "linkerd", "service mesh", "kubectl", "kustomize", "argocd", "flux"]
---

# Kubernetes Specialist

Expert Kubernetes orchestration for cloud-native applications with production-grade deployments.

## Purpose

Comprehensive Kubernetes expertise including Helm charts, custom operators, service mesh, auto-scaling, and GitOps. Ensures K8s deployments are resilient, secure, observable, and cost-effective.

## When to Use

- Deploying microservices to Kubernetes
- Creating Helm charts for reusable deployments
- Implementing auto-scaling (HPA, VPA, Cluster Autoscaler)
- Setting up service mesh for advanced networking
- Building custom operators with Operator SDK
- Implementing GitOps with ArgoCD or Flux
- Optimizing pod scheduling and resource allocation

## Prerequisites

**Required**: Docker, kubectl, basic K8s concepts (Pods, Services, Deployments)

**Agents**: `system-architect`, `cicd-engineer`, `perf-analyzer`, `security-manager`

## Core Workflows

### Workflow 1: Production-Grade Deployment

**Step 1: Create Deployment Manifest**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  labels:
    app: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
        version: v1
    spec:
      containers:
      - name: app
        image: myregistry/my-app:v1.0.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "500m"
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
        securityContext:
          runAsNonRoot: true
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - my-app
              topologyKey: kubernetes.io/hostname
```

**Step 2: Create Service and Ingress**

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app
spec:
  selector:
    app: my-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-app
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - my-app.example.com
    secretName: my-app-tls
  rules:
  - host: my-app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: my-app
            port:
              number: 80
```

### Workflow 2: Helm Chart Development

**Step 1: Create Helm Chart**

```bash
helm create my-app
cd my-app
```

**Step 2: Define Values.yaml**

```yaml
# values.yaml
replicaCount: 3

image:
  repository: myregistry/my-app
  tag: "v1.0.0"
  pullPolicy: IfNotPresent

resources:
  requests:
    memory: "128Mi"
    cpu: "100m"
  limits:
    memory: "256Mi"
    cpu: "500m"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: my-app.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: my-app-tls
      hosts:
        - my-app.example.com
```

**Step 3: Template Deployment**

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "my-app.fullname" . }}
  labels:
    {{- include "my-app.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "my-app.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "my-app.selectorLabels" . | nindent 8 }}
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        ports:
        - containerPort: 8080
        resources:
          {{- toYaml .Values.resources | nindent 12 }}
```

**Step 4: Install Chart**

```bash
helm install my-app . -n production --create-namespace
helm upgrade my-app . -n production
```

### Workflow 3: Horizontal Pod Autoscaler (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
```

### Workflow 4: Service Mesh with Istio

**Step 1: Install Istio**

```bash
curl -L https://istio.io/downloadIstio | sh -
cd istio-*
export PATH=$PWD/bin:$PATH
istioctl install --set profile=demo -y
kubectl label namespace default istio-injection=enabled
```

**Step 2: Define VirtualService for Traffic Management**

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: my-app
spec:
  hosts:
  - my-app
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: my-app
        subset: v2
      weight: 100
  - route:
    - destination:
        host: my-app
        subset: v1
      weight: 90
    - destination:
        host: my-app
        subset: v2
      weight: 10
```

**Step 3: Define DestinationRule**

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: my-app
spec:
  host: my-app
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
```

## Best Practices

**1. Resource Limits Always**
```yaml
# ✅ GOOD
resources:
  requests:
    memory: "128Mi"
    cpu: "100m"
  limits:
    memory: "256Mi"
    cpu: "500m"

# ❌ BAD: No limits (can consume all node resources)
resources: {}
```

**2. Health Probes**
```yaml
# ✅ GOOD: Both liveness and readiness
livenessProbe:
  httpGet:
    path: /health
    port: 8080
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
```

**3. Security Contexts**
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
```

**4. Pod Disruption Budgets**
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: my-app-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: my-app
```

**5. Network Policies**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: my-app-network-policy
spec:
  podSelector:
    matchLabels:
      app: my-app
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
```

## Quality Criteria

- ✅ All workloads have resource limits
- ✅ Health probes configured
- ✅ Security contexts applied
- ✅ PodDisruptionBudgets for critical apps
- ✅ Network policies restrict traffic
- ✅ Secrets managed with external-secrets or sealed-secrets
- ✅ Cluster autoscaling enabled

## Troubleshooting

**Issue**: Pods stuck in Pending
**Solution**: Check node capacity (`kubectl describe node`), review resource requests

**Issue**: Service not reachable
**Solution**: Verify selector matches pod labels, check NetworkPolicies

**Issue**: High memory usage (OOMKilled)
**Solution**: Increase memory limits, profile application memory usage

## Related Skills

- `docker-containerization`: Container best practices
- `aws-specialist`: EKS deployment
- `terraform-iac`: K8s cluster provisioning
- `opentelemetry-observability`: Distributed tracing in K8s

## Tools

- kubectl: Kubernetes CLI
- Helm: Package manager
- Kustomize: YAML templating
- ArgoCD/Flux: GitOps
- Istio/Linkerd: Service mesh
- Lens: Desktop GUI

## MCP Tools

- `mcp__flow-nexus__sandbox_execute` for kubectl commands
- `mcp__memory-mcp__memory_store` for K8s patterns

## Success Metrics

- Deployment time: <5 minutes
- Pod startup time: <30 seconds
- Cluster utilization: 60-80%
- Zero-downtime deployments: 100%

---

**Skill Version**: 1.0.0
**Last Updated**: 2025-11-02
