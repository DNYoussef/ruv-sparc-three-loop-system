---
name: infrastructure
description: '- **Skill ID**: infrastructure-orchestration'
version: 1.0.0
category: operations
tags:
- operations
- deployment
- infrastructure
author: ruv
---

# Infrastructure Orchestration Skill

## Metadata
- **Skill ID**: infrastructure-orchestration
- **Category**: Infrastructure & DevOps
- **Tier**: Gold
- **Version**: 2.0.0
- **Last Updated**: 2025-11-02

## Overview

Comprehensive infrastructure orchestration skill that manages cloud resources, containerization, infrastructure as code (IaC), deployment automation, and monitoring setup. This parent skill coordinates specialized sub-skills for Docker containerization and Terraform IaC management.

## Capabilities

### Core Infrastructure Management
- **Cloud Provisioning**: Multi-cloud resource provisioning (AWS, Azure, GCP)
- **Container Orchestration**: Docker, Kubernetes, Docker Swarm
- **Infrastructure as Code**: Terraform, CloudFormation, Pulumi
- **Configuration Management**: Ansible, Chef, Puppet
- **Deployment Automation**: CI/CD pipelines, blue-green deployments
- **Monitoring & Observability**: Prometheus, Grafana, ELK stack, OpenTelemetry

### Specialized Sub-Skills
1. **Docker Containerization** (`docker-containerization/`)
   - Multi-stage builds, optimization, security scanning
   - Docker Compose orchestration
   - Registry management and image distribution

2. **Terraform IaC** (`terraform-iac/`)
   - Multi-cloud infrastructure provisioning
   - State management and GitOps workflows
   - Module development and reusability

## Trigger Conditions

Auto-invoke this skill when user mentions:
- "infrastructure", "cloud setup", "provision resources"
- "deploy to AWS/Azure/GCP", "multi-cloud"
- "container orchestration", "Kubernetes", "K8s"
- "infrastructure as code", "IaC", "Terraform", "CloudFormation"
- "monitoring setup", "observability", "logging"
- "configuration management", "Ansible"
- "CI/CD pipeline", "deployment automation"

## Agent Assignments

**Primary Agents**:
- `cicd-engineer` - CI/CD pipeline setup and deployment automation
- `backend-dev` - Infrastructure architecture and design
- `system-architect` - High-level infrastructure planning

**Supporting Agents**:
- `code-analyzer` - Infrastructure code review and optimization
- `reviewer` - Security and compliance validation
- `tester` - Infrastructure testing and validation

## Workflow

### 1. Assessment Phase
```yaml
Input: Infrastructure requirements, scale, compliance needs
Actions:
  - Analyze current infrastructure state
  - Identify gaps and requirements
  - Select appropriate tools and platforms
  - Design architecture with redundancy and scalability
Output: Infrastructure design document, technology stack selection
```

### 2. Provisioning Phase
```yaml
Input: Architecture design, resource specifications
Actions:
  - Write IaC templates (Terraform/CloudFormation)
  - Configure networking, security groups, IAM roles
  - Set up container orchestration (if needed)
  - Implement multi-region/AZ deployment
Output: IaC codebase, provisioned cloud resources
```

### 3. Deployment Automation Phase
```yaml
Input: Application artifacts, deployment strategy
Actions:
  - Configure CI/CD pipelines (GitHub Actions, GitLab CI, Jenkins)
  - Set up container registries and artifact storage
  - Implement blue-green or canary deployment strategies
  - Configure auto-scaling and load balancing
Output: Automated deployment pipeline, deployment scripts
```

### 4. Monitoring & Observability Phase
```yaml
Input: SLOs, SLIs, alerting requirements
Actions:
  - Deploy monitoring stack (Prometheus, Grafana, ELK)
  - Configure metrics collection and log aggregation
  - Set up distributed tracing (Jaeger, Zipkin)
  - Create dashboards and alerting rules
Output: Monitoring infrastructure, dashboards, alert configurations
```

### 5. Configuration Management Phase
```yaml
Input: Server configurations, application configs
Actions:
  - Write Ansible playbooks or Chef recipes
  - Implement configuration drift detection
  - Set up secrets management (Vault, AWS Secrets Manager)
  - Configure environment-specific variables
Output: Configuration management codebase, secrets vault
```

### 6. Testing & Validation Phase
```yaml
Input: Infrastructure code, deployment artifacts
Actions:
  - Run infrastructure tests (Terratest, InSpec)
  - Validate security compliance (Checkov, tfsec)
  - Test disaster recovery procedures
  - Perform load testing and chaos engineering
Output: Test reports, compliance validation, DR runbooks
```

## Resource Requirements

### Scripts
- `infra-provisioner.sh` - Automated infrastructure provisioning
- `config-manager.py` - Configuration management and validation
- `deployment-automation.js` - CI/CD pipeline orchestration
- `monitoring-setup.py` - Monitoring stack deployment

### Templates
- `terraform-config.tf` - Terraform configuration for multi-cloud
- `docker-compose.yml` - Docker Compose orchestration
- `k8s-deployment.yaml` - Kubernetes deployment manifests

### Tests
- `infrastructure.test.js` - Infrastructure validation tests
- `deployment.test.py` - Deployment pipeline tests
- `monitoring.test.sh` - Monitoring stack verification

## Integration Points

### Cloud Providers
- **AWS**: EC2, ECS, EKS, Lambda, RDS, S3, CloudFormation
- **Azure**: VMs, AKS, Azure Functions, Cosmos DB, ARM templates
- **GCP**: Compute Engine, GKE, Cloud Functions, Cloud SQL, Deployment Manager

### Container Platforms
- **Docker**: Container runtime, Docker Compose
- **Kubernetes**: Pod orchestration, Helm charts, Operators
- **Nomad**: Alternative orchestration platform

### IaC Tools
- **Terraform**: Multi-cloud provisioning, state management
- **Pulumi**: Programming language-based IaC
- **CloudFormation**: AWS-native IaC
- **Ansible**: Configuration management and provisioning

### Monitoring Tools
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **ELK Stack**: Log aggregation (Elasticsearch, Logstash, Kibana)
- **OpenTelemetry**: Distributed tracing and observability

## Performance Targets

- **Provisioning Time**: < 10 minutes for standard environments
- **Deployment Frequency**: Support multiple deployments per day
- **Recovery Time Objective (RTO)**: < 1 hour for critical systems
- **Recovery Point Objective (RPO)**: < 15 minutes data loss
- **Infrastructure Uptime**: 99.9% SLA
- **Monitoring Coverage**: 100% of critical services

## Best Practices

### Security
- ✅ Implement principle of least privilege (IAM, RBAC)
- ✅ Encrypt data at rest and in transit
- ✅ Use secrets management (never commit secrets)
- ✅ Regular security scanning (Trivy, Checkov, tfsec)
- ✅ Network segmentation and firewall rules

### Reliability
- ✅ Multi-AZ/region deployments for HA
- ✅ Implement health checks and auto-recovery
- ✅ Regular disaster recovery drills
- ✅ Automated backups with tested restore procedures
- ✅ Chaos engineering to test resilience

### Cost Optimization
- ✅ Right-size instances based on actual usage
- ✅ Use spot instances for non-critical workloads
- ✅ Implement auto-scaling policies
- ✅ Regular cost analysis and optimization
- ✅ Tag resources for cost allocation

### Maintainability
- ✅ Infrastructure as Code for all resources
- ✅ Version control for IaC and configuration
- ✅ Comprehensive documentation and runbooks
- ✅ Modular and reusable code
- ✅ Automated testing and validation

## Examples

### Example 1: Docker Deployment
```yaml
Scenario: Deploy microservices application with Docker Compose
Input: Application code, service dependencies
Steps:
  1. Create multi-stage Dockerfiles for each service
  2. Write docker-compose.yml with service definitions
  3. Configure networking and volumes
  4. Set up environment variables and secrets
  5. Implement health checks and restart policies
Output: Production-ready Docker Compose deployment
File: examples/docker-deployment-example.md
```

### Example 2: Kubernetes Setup
```yaml
Scenario: Set up production Kubernetes cluster with monitoring
Input: Application containers, scaling requirements
Steps:
  1. Provision managed K8s cluster (EKS/GKE/AKS)
  2. Create namespaces, deployments, services
  3. Configure Ingress controllers and load balancers
  4. Set up Helm charts for application deployment
  5. Deploy Prometheus/Grafana monitoring stack
  6. Configure auto-scaling (HPA, VPA, cluster autoscaler)
Output: Production K8s cluster with full observability
File: examples/kubernetes-setup-example.md
```

### Example 3: Terraform Multi-Cloud Infrastructure
```yaml
Scenario: Deploy multi-cloud infrastructure with Terraform
Input: Infrastructure requirements, compliance constraints
Steps:
  1. Design multi-cloud architecture (AWS + Azure)
  2. Write Terraform modules for networking, compute, storage
  3. Configure remote state backend (S3 + DynamoDB)
  4. Implement CI/CD pipeline for infrastructure changes
  5. Set up monitoring and alerting across clouds
  6. Run compliance checks (Checkov, tfsec)
Output: Multi-cloud infrastructure with GitOps workflow
File: examples/terraform-infrastructure-example.md
```

## Error Handling

### Common Issues
1. **State Lock Conflicts** (Terraform)
   - Solution: Implement state locking with DynamoDB/Azure Blob
   - Use `-lock=false` only in emergency

2. **Resource Quota Exceeded**
   - Solution: Request quota increases, optimize resource usage
   - Implement cost alerts and limits

3. **Deployment Failures**
   - Solution: Implement rollback strategies, health checks
   - Use blue-green or canary deployments

4. **Configuration Drift**
   - Solution: Regular drift detection, automated remediation
   - Use tools like Terraform Cloud or Spacelift

5. **Secret Leaks**
   - Solution: Use secrets management, never commit secrets
   - Implement pre-commit hooks with tools like git-secrets

## Dependencies

### Required Tools
- Docker (>= 20.10)
- Kubernetes CLI (`kubectl`) (>= 1.24)
- Terraform (>= 1.5)
- Ansible (>= 2.12)
- Cloud CLIs (AWS CLI, Azure CLI, gcloud)

### Optional Tools
- Helm (>= 3.10)
- Terragrunt (>= 0.45)
- Pulumi (>= 3.60)
- Packer (>= 1.8)

## Success Metrics

- ✅ Infrastructure provisioning automated (100% IaC coverage)
- ✅ Zero-downtime deployments achieved
- ✅ Monitoring covers all critical services
- ✅ RTO/RPO targets met in DR tests
- ✅ Security compliance validated (CIS benchmarks)
- ✅ Cost optimization targets achieved

## Related Skills

- `docker-containerization` - Docker-specific orchestration
- `terraform-iac` - Terraform infrastructure management
- `kubernetes-specialist` - Advanced K8s operations
- `aws-specialist` - AWS-specific deployments
- `opentelemetry-observability` - Observability setup
- `cicd-intelligent-recovery` - CI/CD automation

## Notes

- Always use IaC for infrastructure changes (no manual changes)
- Implement tagging strategy for resource organization
- Regular security audits and compliance checks
- Document all runbooks and disaster recovery procedures
- Use multi-region deployments for critical systems
- Implement cost allocation and optimization strategies

---

**Status**: Gold Tier - Production Ready with Comprehensive Resources
**Maintainer**: Infrastructure & DevOps Team
**Support**: Refer to sub-skills for specialized guidance
