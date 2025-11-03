# Infrastructure Orchestration Skill

**Tier**: Gold | **Version**: 2.0.0 | **Category**: Infrastructure & DevOps

## Quick Start

This skill provides comprehensive infrastructure orchestration capabilities including cloud provisioning, container orchestration, IaC management, deployment automation, and monitoring setup.

### Auto-Trigger Keywords
- "infrastructure", "cloud setup", "provision resources"
- "Kubernetes", "K8s", "container orchestration"
- "Terraform", "IaC", "infrastructure as code"
- "monitoring", "observability", "CI/CD pipeline"

## Core Capabilities

### 🏗️ Infrastructure Management
- Multi-cloud provisioning (AWS, Azure, GCP)
- Infrastructure as Code (Terraform, CloudFormation)
- Container orchestration (Docker, Kubernetes)
- Configuration management (Ansible)

### 🚀 Deployment Automation
- CI/CD pipeline setup
- Blue-green and canary deployments
- Auto-scaling and load balancing
- Artifact management

### 📊 Monitoring & Observability
- Prometheus/Grafana stack
- ELK stack for log aggregation
- Distributed tracing (Jaeger, OpenTelemetry)
- Custom dashboards and alerting

## Quick Usage Examples

### Docker Deployment
```bash
# Use the docker-containerization sub-skill
# See: docker-containerization/skill.md
# Automatically handles multi-stage builds, optimization, security scanning
```

### Kubernetes Setup
```bash
# Full production K8s cluster with monitoring
# Provisions managed cluster, deploys apps, sets up observability
# Refer to: examples/kubernetes-setup-example.md
```

### Terraform Infrastructure
```bash
# Multi-cloud infrastructure with GitOps
# Uses terraform-iac sub-skill for provisioning
# See: terraform-iac/skill.md and examples/terraform-infrastructure-example.md
```

## File Structure

```
infrastructure/
├── skill.md                          # Main skill documentation
├── README.md                         # This file
├── resources/
│   ├── scripts/
│   │   ├── infra-provisioner.sh      # Infrastructure provisioning automation
│   │   ├── config-manager.py         # Configuration management
│   │   ├── deployment-automation.js  # CI/CD orchestration
│   │   └── monitoring-setup.py       # Monitoring stack deployment
│   └── templates/
│       ├── terraform-config.tf       # Terraform multi-cloud template
│       ├── docker-compose.yml        # Docker Compose template
│       └── k8s-deployment.yaml       # Kubernetes deployment template
├── tests/
│   ├── infrastructure.test.js        # Infrastructure validation
│   ├── deployment.test.py            # Deployment pipeline tests
│   └── monitoring.test.sh            # Monitoring stack tests
├── examples/
│   ├── docker-deployment-example.md       # Docker microservices deployment
│   ├── kubernetes-setup-example.md        # Production K8s cluster setup
│   └── terraform-infrastructure-example.md # Multi-cloud IaC deployment
├── docker-containerization/          # Sub-skill for Docker
└── terraform-iac/                    # Sub-skill for Terraform
```

## Agent Assignments

**Primary**: `cicd-engineer`, `backend-dev`, `system-architect`
**Supporting**: `code-analyzer`, `reviewer`, `tester`

## Workflow Phases

1. **Assessment** - Analyze requirements, design architecture
2. **Provisioning** - Deploy cloud resources with IaC
3. **Deployment Automation** - Set up CI/CD pipelines
4. **Monitoring** - Deploy observability stack
5. **Configuration** - Manage configs and secrets
6. **Testing** - Validate infrastructure and deployments

## Performance Targets

- Provisioning: < 10 minutes
- Deployment frequency: Multiple per day
- RTO: < 1 hour
- RPO: < 15 minutes
- Uptime: 99.9% SLA

## Best Practices

✅ Infrastructure as Code for all resources
✅ Multi-AZ/region for high availability
✅ Secrets management (never commit secrets)
✅ Regular security scanning and compliance
✅ Automated testing and disaster recovery drills
✅ Cost optimization and resource tagging

## Resources

### Scripts
- **infra-provisioner.sh** - Automates cloud resource provisioning
- **config-manager.py** - Validates and manages configurations
- **deployment-automation.js** - Orchestrates CI/CD pipelines
- **monitoring-setup.py** - Deploys monitoring infrastructure

### Templates
- **terraform-config.tf** - Multi-cloud Terraform configuration
- **docker-compose.yml** - Microservices orchestration
- **k8s-deployment.yaml** - Kubernetes deployment manifests

### Tests
- **infrastructure.test.js** - Infrastructure validation (Terratest)
- **deployment.test.py** - Deployment pipeline verification
- **monitoring.test.sh** - Monitoring stack health checks

## Examples

### 1. Docker Deployment (150-300 lines)
Complete microservices deployment with Docker Compose including:
- Multi-stage Dockerfiles for optimization
- Service orchestration and networking
- Environment variables and secrets management
- Health checks and auto-restart policies

### 2. Kubernetes Setup (150-300 lines)
Production K8s cluster with full observability:
- Managed cluster provisioning (EKS/GKE/AKS)
- Deployments, services, ingress controllers
- Helm charts and auto-scaling
- Prometheus/Grafana monitoring stack

### 3. Terraform Infrastructure (150-300 lines)
Multi-cloud infrastructure deployment:
- AWS + Azure architecture
- Terraform modules for networking, compute, storage
- Remote state backend with locking
- CI/CD pipeline for infrastructure changes
- Compliance scanning (Checkov, tfsec)

## Integration Points

**Cloud**: AWS, Azure, GCP
**Containers**: Docker, Kubernetes, Nomad
**IaC**: Terraform, Pulumi, CloudFormation, Ansible
**Monitoring**: Prometheus, Grafana, ELK, OpenTelemetry
**CI/CD**: GitHub Actions, GitLab CI, Jenkins

## Related Skills

- `docker-containerization` - Docker-specific operations
- `terraform-iac` - Terraform infrastructure management
- `kubernetes-specialist` - Advanced K8s orchestration
- `aws-specialist` - AWS-specific deployments
- `opentelemetry-observability` - Observability stack
- `cicd-intelligent-recovery` - CI/CD automation

## Support

For detailed documentation, refer to:
- `skill.md` - Complete skill specification
- Sub-skill directories for specialized guidance
- Example files for hands-on walkthroughs

---

**Status**: Gold Tier - Production Ready
**Last Updated**: 2025-11-02
**Maintainer**: Infrastructure & DevOps Team
