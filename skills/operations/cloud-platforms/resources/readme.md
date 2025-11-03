# Cloud Platforms Resources

This directory contains production-ready automation scripts and infrastructure templates for multi-cloud deployments.

## Directory Structure

```
resources/
├── scripts/          # Deployment automation scripts
│   ├── deploy_aws.py         # AWS deployment automation (Python)
│   ├── deploy_k8s.sh         # Kubernetes deployment (Bash)
│   ├── terraform_apply.py    # Terraform automation (Python)
│   └── gcp_deploy.sh         # GCP deployment (Bash)
└── templates/        # Infrastructure as Code templates
    ├── aws-infra.tf          # AWS Terraform configuration
    ├── k8s-deployment.yaml   # Kubernetes production manifest
    ├── gcp-config.json       # GCP deployment configuration
    └── docker-compose.yaml   # Local development stack
```

## Scripts Overview

### deploy_aws.py
AWS deployment automation supporting Lambda, ECS, CloudFormation, and EC2.

**Commands:**
- `lambda` - Deploy Lambda functions
- `ecs` - Deploy ECS Fargate services
- `cfn` - Deploy CloudFormation stacks
- `ec2` - Deploy EC2 instances

**Example:**
```bash
python deploy_aws.py lambda --name my-func --zip function.zip --handler index.handler
```

### deploy_k8s.sh
Kubernetes deployment automation with kubectl and Helm support.

**Commands:**
- `kubectl` - Deploy using kubectl
- `helm` - Deploy Helm charts
- `image` - Deploy from Docker image
- `scale` - Scale deployments
- `rollback` - Rollback deployments
- `status` - Get deployment status

**Example:**
```bash
./deploy_k8s.sh helm myapp ./charts/app production values.yaml
```

### terraform_apply.py
Terraform workflow automation with state management.

**Commands:**
- `init` - Initialize Terraform
- `validate` - Validate configuration
- `fmt` - Format Terraform files
- `plan` - Create execution plan
- `apply` - Apply changes
- `destroy` - Destroy infrastructure
- `output` - Get outputs
- `state` - List state resources

**Example:**
```bash
python terraform_apply.py plan --var-file prod.tfvars --out tfplan
python terraform_apply.py apply --plan-file tfplan
```

### gcp_deploy.sh
GCP deployment automation for Cloud Run, Cloud Functions, GKE, and Compute Engine.

**Commands:**
- `cloud-run` - Deploy Cloud Run services
- `function` - Deploy Cloud Functions
- `gke` - Deploy to GKE cluster
- `create-gke` - Create GKE cluster
- `compute` - Deploy Compute Engine instances
- `build` - Build and push Docker images
- `app-engine` - Deploy App Engine applications

**Example:**
```bash
./gcp_deploy.sh cloud-run myservice gcr.io/project/image us-central1
```

## Templates Overview

### aws-infra.tf
Complete AWS infrastructure with:
- VPC with public/private subnets
- Application Load Balancer
- ECS Fargate cluster
- RDS PostgreSQL database
- CloudWatch logging
- IAM roles and security groups

**Usage:**
```bash
terraform init
terraform plan -var="project_name=myapp" -var="container_image=nginx:latest"
terraform apply
```

### k8s-deployment.yaml
Production Kubernetes deployment with:
- Deployment with 3 replicas
- Service (ClusterIP)
- Ingress with TLS
- HorizontalPodAutoscaler
- ConfigMap and Secrets
- Health probes
- Resource limits

**Usage:**
```bash
kubectl apply -f k8s-deployment.yaml
```

### gcp-config.json
Comprehensive GCP configuration for:
- Cloud Run services
- Cloud Functions
- GKE clusters
- Compute Engine
- Cloud SQL
- Cloud Storage
- Monitoring and alerting

**Usage:**
Use as reference for GCP deployment scripts or programmatic API calls.

### docker-compose.yaml
Full local development environment with:
- Web application
- PostgreSQL database
- Redis cache
- Nginx reverse proxy
- Prometheus + Grafana
- Database and cache UIs
- Background workers

**Usage:**
```bash
docker-compose up -d
docker-compose ps
docker-compose logs -f web
```

## Prerequisites

### AWS Scripts
- AWS CLI configured
- Python 3.11+
- boto3 library: `pip install boto3`

### Kubernetes Scripts
- kubectl installed
- Helm 3+ installed
- Access to Kubernetes cluster

### Terraform Scripts
- Terraform 1.0+ installed
- Python 3.11+

### GCP Scripts
- gcloud CLI configured
- Authenticated GCP account
- Bash shell

## Quick Start

1. **Clone or copy templates to your project:**
   ```bash
   cp templates/aws-infra.tf ./infrastructure/
   cp templates/k8s-deployment.yaml ./kubernetes/
   ```

2. **Make scripts executable:**
   ```bash
   chmod +x scripts/*.sh
   ```

3. **Deploy infrastructure:**
   ```bash
   # AWS
   cd infrastructure
   python ../scripts/terraform_apply.py apply

   # Kubernetes
   ./scripts/deploy_k8s.sh kubectl kubernetes/k8s-deployment.yaml production

   # GCP
   ./scripts/gcp_deploy.sh cloud-run myapp gcr.io/project/myapp us-central1
   ```

## Best Practices

1. **Version Control:** Commit infrastructure templates to Git
2. **Variables:** Use .tfvars files for environment-specific configs
3. **Secrets:** Store secrets in AWS Secrets Manager, GCP Secret Manager, or K8s Secrets
4. **State Management:** Use remote state for Terraform (S3, GCS)
5. **Testing:** Test in dev/staging before production
6. **Monitoring:** Enable CloudWatch, Stackdriver, Prometheus
7. **Backup:** Regular automated backups of databases
8. **Documentation:** Keep runbooks up to date

## Troubleshooting

### Script Execution Errors
```bash
# Check dependencies
python --version  # Should be 3.11+
aws --version
gcloud --version
kubectl version
terraform version

# Verify authentication
aws sts get-caller-identity
gcloud auth list
kubectl cluster-info
```

### Common Issues

**AWS: Lambda deployment fails**
- Check IAM role permissions
- Verify zip file size (<50 MB)
- Ensure handler path is correct

**Kubernetes: Deployment stuck in Pending**
- Check resource quotas: `kubectl describe quota -n production`
- Verify PVC availability: `kubectl get pvc -n production`
- Check node capacity: `kubectl top nodes`

**Terraform: State lock error**
- Check DynamoDB table for locks
- Manually release lock if needed: `terraform force-unlock LOCK_ID`

**GCP: Cloud Run deployment fails**
- Verify service account permissions
- Check container image exists in GCR
- Ensure region supports Cloud Run

## Support

For issues or questions:
1. Check test suites in `../tests/` for examples
2. Review enhancement summary: `../ENHANCEMENT-SUMMARY.md`
3. Consult cloud provider documentation

## License

MIT License - See main repository for details
