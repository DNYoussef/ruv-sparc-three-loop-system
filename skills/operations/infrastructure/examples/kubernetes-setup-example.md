# Production Kubernetes Cluster Setup Example

**Scenario**: Set up a production-grade Kubernetes cluster on AWS EKS with auto-scaling, monitoring (Prometheus/Grafana), logging (ELK), ingress controller, cert-manager for TLS, and comprehensive observability.

**Components**: EKS cluster, Application deployment, Horizontal Pod Autoscaler, Prometheus/Grafana stack, Ingress with TLS, Network policies, Pod Disruption Budgets

**Lines**: ~280

---

## Prerequisites

```bash
# Required tools
aws --version          # AWS CLI 2.0+
kubectl version        # kubectl 1.24+
eksctl version         # eksctl 0.140+
helm version           # Helm 3.10+
```

## Step 1: Provision EKS Cluster with eksctl

Create `cluster-config.yaml`:

```yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: production-cluster
  region: us-east-1
  version: "1.27"
  tags:
    Environment: production
    ManagedBy: eksctl

# VPC Configuration
vpc:
  cidr: 10.0.0.0/16
  nat:
    gateway: HighlyAvailable  # One NAT gateway per AZ
  clusterEndpoints:
    publicAccess: true
    privateAccess: true

# IAM Configuration
iam:
  withOIDC: true
  serviceAccounts:
    - metadata:
        name: cluster-autoscaler
        namespace: kube-system
      wellKnownPolicies:
        autoScaler: true
    - metadata:
        name: ebs-csi-controller-sa
        namespace: kube-system
      wellKnownPolicies:
        ebsCSIController: true
    - metadata:
        name: aws-load-balancer-controller
        namespace: kube-system
      wellKnownPolicies:
        awsLoadBalancerController: true

# Managed Node Groups
managedNodeGroups:
  - name: production-workers
    instanceType: t3.large
    minSize: 3
    maxSize: 10
    desiredCapacity: 5
    volumeSize: 100
    volumeType: gp3
    privateNetworking: true
    ssh:
      allow: true
      publicKeyPath: ~/.ssh/eks-production.pub
    labels:
      role: worker
      environment: production
    tags:
      k8s.io/cluster-autoscaler/enabled: "true"
      k8s.io/cluster-autoscaler/production-cluster: "owned"
    iam:
      withAddonPolicies:
        autoScaler: true
        albIngress: true
        ebs: true
        efs: true
        cloudWatch: true

# CloudWatch Logging
cloudWatch:
  clusterLogging:
    enableTypes:
      - api
      - audit
      - authenticator
      - controllerManager
      - scheduler
    logRetentionInDays: 30

# Add-ons
addons:
  - name: vpc-cni
    version: latest
  - name: coredns
    version: latest
  - name: kube-proxy
    version: latest
  - name: aws-ebs-csi-driver
    version: latest
```

Create cluster:

```bash
# Provision EKS cluster (takes 15-20 minutes)
eksctl create cluster -f cluster-config.yaml

# Verify cluster
kubectl cluster-info
kubectl get nodes

# Enable kubectl autocomplete
echo 'source <(kubectl completion bash)' >> ~/.bashrc
source ~/.bashrc
```

## Step 2: Install Cluster Autoscaler

```bash
# Download autoscaler manifest
curl -o cluster-autoscaler-autodiscover.yaml \
  https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml

# Edit manifest - add cluster name
sed -i "s/<YOUR CLUSTER NAME>/production-cluster/g" cluster-autoscaler-autodiscover.yaml

# Apply manifest
kubectl apply -f cluster-autoscaler-autodiscover.yaml

# Verify autoscaler is running
kubectl -n kube-system get pods -l app=cluster-autoscaler
kubectl -n kube-system logs -l app=cluster-autoscaler
```

## Step 3: Install AWS Load Balancer Controller

```bash
# Add EKS Helm repository
helm repo add eks https://aws.github.io/eks-charts
helm repo update

# Install AWS Load Balancer Controller
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=production-cluster \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller

# Verify installation
kubectl -n kube-system get pods -l app.kubernetes.io/name=aws-load-balancer-controller
```

## Step 4: Install cert-manager for TLS

```bash
# Install cert-manager CRDs
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Wait for cert-manager to be ready
kubectl -n cert-manager wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager --timeout=90s

# Create ClusterIssuer for Let's Encrypt
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: nginx
EOF

# Verify ClusterIssuer
kubectl get clusterissuer letsencrypt-prod
```

## Step 5: Install Nginx Ingress Controller

```bash
# Add Nginx Helm repository
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

# Install Nginx Ingress Controller
helm install nginx-ingress ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.replicaCount=3 \
  --set controller.nodeSelector."kubernetes\.io/os"=linux \
  --set controller.admissionWebhooks.patch.nodeSelector."kubernetes\.io/os"=linux \
  --set controller.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-type"="nlb" \
  --set controller.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-cross-zone-load-balancing-enabled"="true" \
  --set controller.metrics.enabled=true \
  --set controller.metrics.serviceMonitor.enabled=true

# Wait for load balancer to be provisioned
kubectl -n ingress-nginx get service nginx-ingress-ingress-nginx-controller -w
```

## Step 6: Deploy Prometheus Stack

```bash
# Add Prometheus community Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Create namespace
kubectl create namespace monitoring

# Install kube-prometheus-stack (Prometheus + Grafana + Alertmanager)
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=gp3 \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
  --set grafana.enabled=true \
  --set grafana.adminPassword=admin \
  --set grafana.persistence.enabled=true \
  --set grafana.persistence.storageClassName=gp3 \
  --set grafana.persistence.size=10Gi \
  --set alertmanager.enabled=true

# Verify Prometheus stack
kubectl -n monitoring get pods
kubectl -n monitoring get svc
```

## Step 7: Deploy Application with Full Configuration

The Kubernetes manifests are already provided in `resources/templates/k8s-deployment.yaml`.

Apply the manifests:

```bash
# Create namespace
kubectl create namespace production

# Create ConfigMap
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: production
data:
  NODE_ENV: "production"
  LOG_LEVEL: "info"
  PORT: "3000"
  API_URL: "https://api.example.com"
EOF

# Create Secret
kubectl create secret generic app-secrets \
  --namespace production \
  --from-literal=JWT_SECRET=$(openssl rand -base64 32) \
  --from-literal=DATABASE_PASSWORD=$(openssl rand -base64 32)

# Apply all manifests
kubectl apply -f resources/templates/k8s-deployment.yaml

# Verify deployment
kubectl -n production get all
kubectl -n production get ingress
```

## Step 8: Configure Horizontal Pod Autoscaler

The HPA is already configured in the manifests. Verify it:

```bash
# Check HPA status
kubectl -n production get hpa

# Describe HPA for details
kubectl -n production describe hpa api-hpa

# Watch HPA scaling
kubectl -n production get hpa api-hpa -w
```

## Step 9: Install Metrics Server (if not already installed)

```bash
# Install metrics-server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Verify metrics-server
kubectl -n kube-system get pods -l k8s-app=metrics-server

# Test metrics collection
kubectl top nodes
kubectl top pods -n production
```

## Step 10: Deploy ELK Stack for Logging

Create `elk-values.yaml`:

```yaml
# Elasticsearch configuration
elasticsearch:
  replicas: 3
  minimumMasterNodes: 2
  resources:
    requests:
      cpu: 1000m
      memory: 2Gi
    limits:
      cpu: 2000m
      memory: 4Gi
  volumeClaimTemplate:
    accessModes: ["ReadWriteOnce"]
    storageClassName: gp3
    resources:
      requests:
        storage: 100Gi
  esJavaOpts: "-Xmx2g -Xms2g"

# Kibana configuration
kibana:
  replicas: 1
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 1000m
      memory: 2Gi
  ingress:
    enabled: true
    className: nginx
    annotations:
      cert-manager.io/cluster-issuer: letsencrypt-prod
    hosts:
      - host: kibana.example.com
        paths:
          - path: /
            pathType: Prefix
    tls:
      - secretName: kibana-tls
        hosts:
          - kibana.example.com

# Logstash configuration
logstash:
  replicas: 2
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 1000m
      memory: 2Gi
```

Install ELK stack:

```bash
# Add Elastic Helm repository
helm repo add elastic https://helm.elastic.co
helm repo update

# Install Elasticsearch
helm install elasticsearch elastic/elasticsearch \
  --namespace logging \
  --create-namespace \
  -f elk-values.yaml

# Install Kibana
helm install kibana elastic/kibana \
  --namespace logging \
  -f elk-values.yaml

# Install Filebeat for log collection
helm install filebeat elastic/filebeat \
  --namespace logging \
  --set daemonset.enabled=true

# Verify ELK stack
kubectl -n logging get pods
```

## Step 11: Configure Ingress with TLS

```bash
# Update Ingress manifest with your domain
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  namespace: production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
    - hosts:
        - app.example.com
        - api.example.com
      secretName: app-tls-cert
  rules:
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend
                port:
                  number: 80
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api
                port:
                  number: 3000
EOF

# Wait for TLS certificate
kubectl -n production get certificate app-tls-cert -w

# Verify Ingress
kubectl -n production describe ingress app-ingress

# Get load balancer URL
kubectl -n ingress-nginx get svc nginx-ingress-ingress-nginx-controller
```

## Step 12: Configure Network Policies

Network policies are already included in the manifests. Verify them:

```bash
# List network policies
kubectl -n production get networkpolicies

# Describe network policy
kubectl -n production describe networkpolicy api-network-policy

# Test network connectivity (from frontend to API should work)
kubectl -n production exec -it deployment/frontend -- nc -zv api 3000

# Test blocked connection (external to API should fail)
kubectl run test-pod --rm -it --image=alpine -- nc -zv api.production.svc.cluster.local 3000
```

## Step 13: Configure Pod Disruption Budgets

PDBs are already configured. Verify them:

```bash
# List PDBs
kubectl -n production get pdb

# Describe PDB
kubectl -n production describe pdb api-pdb

# Test disruption (drain node should respect PDB)
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
```

## Step 14: Access Monitoring Dashboards

```bash
# Port-forward to Grafana
kubectl -n monitoring port-forward svc/prometheus-grafana 3000:80

# Access Grafana at http://localhost:3000
# Username: admin
# Password: admin (or value set during installation)

# Import Kubernetes dashboards:
# - Dashboard ID 315 (Kubernetes cluster monitoring)
# - Dashboard ID 6417 (Kubernetes cluster monitoring (Prometheus))
# - Dashboard ID 1621 (Kubernetes cluster monitoring (via Kube-state-metrics))

# Port-forward to Prometheus
kubectl -n monitoring port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090

# Access Prometheus at http://localhost:9090
```

## Step 15: Configure Backup Strategy

Create backup script for EKS:

```bash
#!/bin/bash
# backup-eks.sh

BACKUP_DIR="./eks-backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup all Kubernetes resources
kubectl get all --all-namespaces -o yaml > "$BACKUP_DIR/all-resources.yaml"

# Backup specific namespaces
for ns in production monitoring logging; do
  kubectl get all -n $ns -o yaml > "$BACKUP_DIR/$ns-resources.yaml"
  kubectl get configmap -n $ns -o yaml > "$BACKUP_DIR/$ns-configmaps.yaml"
  kubectl get secret -n $ns -o yaml > "$BACKUP_DIR/$ns-secrets.yaml"
  kubectl get ingress -n $ns -o yaml > "$BACKUP_DIR/$ns-ingress.yaml"
  kubectl get pvc -n $ns -o yaml > "$BACKUP_DIR/$ns-pvcs.yaml"
done

# Backup Helm releases
helm list --all-namespaces -o yaml > "$BACKUP_DIR/helm-releases.yaml"

echo "Backup completed: $BACKUP_DIR"
```

## Step 16: Monitoring and Observability

```bash
# View application logs
kubectl -n production logs -f deployment/api --tail=100

# View events
kubectl -n production get events --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n production
kubectl top nodes

# View pod details
kubectl -n production describe pod <pod-name>

# Execute commands in pod
kubectl -n production exec -it deployment/api -- /bin/sh

# View HPA metrics
kubectl -n production get hpa api-hpa -o yaml
```

## Step 17: Cluster Upgrade Strategy

```bash
# Check current cluster version
eksctl get cluster production-cluster

# Upgrade control plane
eksctl upgrade cluster --name production-cluster --version 1.28 --approve

# Upgrade node groups
eksctl upgrade nodegroup \
  --name production-workers \
  --cluster production-cluster \
  --kubernetes-version 1.28

# Verify upgrade
kubectl get nodes
kubectl version
```

## Troubleshooting

### Issue: Pods not scheduling
```bash
# Check node resources
kubectl describe nodes

# Check pod events
kubectl -n production describe pod <pod-name>

# Check resource requests vs limits
kubectl -n production get pods -o custom-columns=NAME:.metadata.name,CPU_REQ:.spec.containers[*].resources.requests.cpu,MEM_REQ:.spec.containers[*].resources.requests.memory
```

### Issue: Ingress not working
```bash
# Check Ingress controller logs
kubectl -n ingress-nginx logs deployment/nginx-ingress-ingress-nginx-controller

# Verify Ingress resource
kubectl -n production describe ingress app-ingress

# Check TLS certificate
kubectl -n production describe certificate app-tls-cert
```

### Issue: HPA not scaling
```bash
# Check metrics-server
kubectl -n kube-system logs deployment/metrics-server

# Verify metrics availability
kubectl top pods -n production

# Check HPA details
kubectl -n production describe hpa api-hpa
```

## Performance Optimization

1. **Node selection**: Use appropriate instance types (memory/CPU optimized)
2. **Resource requests/limits**: Right-size based on actual usage
3. **HPA configuration**: Tune CPU/memory thresholds and scaling policies
4. **PDB configuration**: Balance availability with cluster maintenance
5. **Ingress optimization**: Configure connection pooling, timeouts
6. **Persistent volume**: Use gp3 instead of gp2 for better performance

## Cost Optimization

```bash
# Use Spot instances for non-critical workloads
# Add spot instance node group to cluster-config.yaml

managedNodeGroups:
  - name: production-spot
    instanceType: t3.large
    minSize: 2
    maxSize: 10
    spot: true
    labels:
      lifecycle: spot
    taints:
      - key: spot
        value: "true"
        effect: NoSchedule
```

## Security Best Practices

✅ Private subnets for worker nodes
✅ Security groups and network policies
✅ RBAC with least privilege
✅ Pod security policies
✅ Secrets encrypted at rest
✅ Regular security updates
✅ TLS for all external traffic
✅ Service mesh for internal encryption (optional: Istio)

---

**Result**: Production-ready Kubernetes cluster with:
- High availability (multi-AZ)
- Auto-scaling (HPA + Cluster Autoscaler)
- Complete observability (Prometheus + Grafana + ELK)
- TLS encryption (cert-manager + Let's Encrypt)
- Network isolation (Network Policies)
- Graceful disruption handling (PDBs)
- Cost optimization (Spot instances support)
