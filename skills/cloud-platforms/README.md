# Cloud Platforms - Multi-Cloud Infrastructure Management

Comprehensive cloud deployment and infrastructure management across AWS, Google Cloud Platform, and Microsoft Azure. This skill provides production-ready deployment patterns, Infrastructure as Code templates, and best practices for serverless, containerized, and traditional compute workloads.

## Overview

Modern applications require flexible deployment strategies across multiple cloud providers. This skill enables you to:

- **Multi-Cloud Deployment**: Deploy applications to AWS, GCP, and Azure with consistent patterns
- **Infrastructure as Code**: Manage infrastructure using Terraform, CloudFormation, ARM templates
- **Serverless Architecture**: Build and deploy serverless applications with AWS Lambda, Cloud Functions, Azure Functions
- **Container Orchestration**: Deploy containerized workloads with ECS, GKE, AKS
- **Cost Optimization**: Implement cost-effective architectures with reserved instances, spot instances, and auto-scaling

## Quick Start

### Prerequisites

```bash
# Install required CLI tools
aws configure                        # AWS CLI
gcloud init                          # Google Cloud SDK
az login                             # Azure CLI
terraform init                       # Terraform
```

### Basic Deployment Workflow

1. **Define Requirements**
   - Workload type (API, batch processing, streaming)
   - Scaling requirements (vertical, horizontal, auto-scaling)
   - Compliance needs (HIPAA, PCI-DSS, SOC 2)
   - Budget constraints

2. **Select Platform and Services**
   - Choose primary cloud provider based on requirements
   - Identify appropriate services (compute, storage, database)
   - Design for high availability (multi-AZ, multi-region)
   - Plan disaster recovery strategy (RTO, RPO)

3. **Provision Infrastructure**
   - Write Infrastructure as Code (Terraform recommended for multi-cloud)
   - Implement security best practices (IAM, encryption, network isolation)
   - Configure monitoring and logging
   - Set up CI/CD pipelines

4. **Deploy Application**
   - Containerize application with Docker
   - Deploy using platform-specific tools (CloudFormation, Deployment Manager)
   - Implement blue-green or canary deployments
   - Configure auto-scaling policies

5. **Monitor and Optimize**
   - Track resource utilization and costs
   - Optimize performance (caching, CDN, database tuning)
   - Review security posture regularly
   - Implement cost-saving measures

## Supported Platforms

### AWS (Amazon Web Services)

**Compute Services**:
- EC2: Virtual machines with flexible instance types
- Lambda: Serverless functions (Node.js, Python, Go, Java)
- ECS/Fargate: Container orchestration (serverless containers)
- Batch: Large-scale batch processing
- Elastic Beanstalk: Platform-as-a-Service for web apps

**Storage Services**:
- S3: Object storage with 99.999999999% durability
- EBS: Block storage for EC2 instances
- EFS: Elastic file system for shared storage
- Glacier: Long-term archival storage

**Database Services**:
- RDS: Managed relational databases (PostgreSQL, MySQL, MariaDB, Oracle, SQL Server)
- DynamoDB: NoSQL key-value database
- Aurora: High-performance MySQL/PostgreSQL compatible database
- Redshift: Data warehouse for analytics

**Networking**:
- VPC: Virtual private cloud with subnet isolation
- Route 53: DNS and domain management
- CloudFront: Global CDN for content delivery
- API Gateway: RESTful API management

### Google Cloud Platform (GCP)

**Compute Services**:
- Compute Engine: Virtual machines with custom machine types
- Cloud Functions: Event-driven serverless functions
- GKE (Google Kubernetes Engine): Managed Kubernetes clusters
- Cloud Run: Fully managed serverless containers
- App Engine: Platform-as-a-Service for web applications

**Storage Services**:
- Cloud Storage: Object storage with multi-regional replication
- Persistent Disk: Block storage for Compute Engine
- Filestore: Managed NFS file storage

**Database Services**:
- Cloud SQL: Managed relational databases (PostgreSQL, MySQL, SQL Server)
- Firestore: NoSQL document database
- BigQuery: Serverless data warehouse for analytics
- Spanner: Globally distributed relational database

**Networking**:
- VPC: Virtual private cloud with global reach
- Cloud CDN: Content delivery network
- Cloud Load Balancing: Global load balancing
- Cloud DNS: Managed DNS service

### Microsoft Azure

**Compute Services**:
- Virtual Machines: IaaS compute with Windows/Linux support
- Azure Functions: Serverless event-driven compute
- AKS (Azure Kubernetes Service): Managed Kubernetes
- Container Instances: Serverless containers
- App Service: PaaS for web apps and APIs

**Storage Services**:
- Blob Storage: Object storage for unstructured data
- Disk Storage: Persistent disks for VMs
- Azure Files: Managed file shares
- Archive Storage: Long-term cold storage

**Database Services**:
- Azure SQL Database: Managed SQL Server
- Cosmos DB: Globally distributed NoSQL database
- Azure Database for PostgreSQL/MySQL: Managed open-source databases
- Synapse Analytics: Data warehouse and analytics

**Networking**:
- Virtual Network: Software-defined networking
- Traffic Manager: DNS-based load balancing
- Front Door: Global HTTP/HTTPS load balancer
- Azure CDN: Content delivery network

## Multi-Cloud Deployment Strategy

### When to Use Multi-Cloud

- **Avoid Vendor Lock-in**: Maintain flexibility to switch providers
- **Geographic Coverage**: Leverage regional strengths of each provider
- **Cost Optimization**: Use competitive pricing across providers
- **Compliance**: Meet regulatory requirements in different regions
- **Disaster Recovery**: Cross-cloud failover for critical services

### Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Cloud Application                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │     AWS      │  │     GCP      │  │    Azure     │      │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤      │
│  │ • Lambda     │  │ • Cloud Run  │  │ • Functions  │      │
│  │ • ECS        │  │ • GKE        │  │ • AKS        │      │
│  │ • RDS        │  │ • Cloud SQL  │  │ • SQL DB     │      │
│  │ • S3         │  │ • GCS        │  │ • Blob       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          Terraform (Multi-Cloud IaC)                │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │     Monitoring: Datadog, New Relic, Prometheus      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Best Practices

### Security

1. **Identity and Access Management**
   - Use least privilege principle for IAM policies
   - Enable MFA for all user accounts
   - Rotate access keys regularly (90 days)
   - Use service accounts/roles instead of user credentials

2. **Encryption**
   - Enable encryption at rest for all storage (S3, EBS, Cloud Storage)
   - Use TLS/SSL for all data in transit
   - Manage keys with KMS/Cloud KMS/Key Vault
   - Implement certificate rotation

3. **Network Security**
   - Use VPC/VNet isolation for resources
   - Implement security groups/firewall rules
   - Enable VPN or Direct Connect for hybrid cloud
   - Use private subnets for databases and backend services

### Cost Optimization

1. **Compute**
   - Use reserved instances for predictable workloads (up to 75% savings)
   - Leverage spot instances for fault-tolerant workloads (up to 90% savings)
   - Right-size instances based on actual usage
   - Use auto-scaling to match demand

2. **Storage**
   - Implement lifecycle policies for S3/Cloud Storage/Blob
   - Use appropriate storage classes (Standard, Infrequent Access, Archive)
   - Enable compression for data at rest
   - Delete unused snapshots and backups

3. **Database**
   - Use read replicas for read-heavy workloads
   - Implement connection pooling
   - Scale vertically before horizontally
   - Consider serverless database options (Aurora Serverless, Cloud SQL)

### High Availability

1. **Multi-AZ Deployment**
   - Deploy across multiple availability zones (minimum 2)
   - Use load balancers for traffic distribution
   - Implement health checks and auto-failover
   - Test failover procedures regularly

2. **Disaster Recovery**
   - Define RTO (Recovery Time Objective) and RPO (Recovery Point Objective)
   - Implement automated backups with retention policies
   - Use cross-region replication for critical data
   - Document and test disaster recovery procedures

3. **Auto-Scaling**
   - Configure horizontal auto-scaling for web/app tiers
   - Use predictive scaling for known traffic patterns
   - Set appropriate cooldown periods
   - Monitor scaling metrics (CPU, memory, custom metrics)

### Monitoring and Observability

1. **Logging**
   - Centralize logs (CloudWatch Logs, Cloud Logging, Log Analytics)
   - Implement structured logging (JSON format)
   - Set log retention policies
   - Create log-based alerts for critical events

2. **Metrics**
   - Track application metrics (latency, throughput, error rates)
   - Monitor infrastructure metrics (CPU, memory, disk, network)
   - Implement custom metrics for business KPIs
   - Set up dashboards for real-time visibility

3. **Tracing**
   - Implement distributed tracing (X-Ray, Cloud Trace, Application Insights)
   - Track request flows across microservices
   - Identify performance bottlenecks
   - Correlate logs, metrics, and traces

## Examples

This skill includes three comprehensive deployment examples:

1. **[AWS Lambda Serverless API](examples/example-1-aws-lambda.md)**: Build a serverless REST API with Lambda, API Gateway, and DynamoDB
2. **[GCP Cloud Run Container Deployment](examples/example-2-gcp-cloud-run.md)**: Deploy a containerized web application with Cloud Run and Cloud SQL
3. **[Azure Functions Event Processing](examples/example-3-azure-functions.md)**: Build an event-driven processing pipeline with Azure Functions and Cosmos DB

## References

- **[AWS Services Reference](references/aws-services.md)**: Comprehensive guide to AWS compute, storage, database, and networking services
- **[Terraform Infrastructure as Code](references/terraform-iac.md)**: Best practices for multi-cloud infrastructure provisioning with Terraform

## Architecture Diagrams

See [graphviz/workflow.dot](graphviz/workflow.dot) for a visual representation of multi-cloud deployment architecture.

## Integration with SPARC Methodology

This skill integrates seamlessly with the SPARC development workflow:

1. **Specification**: Define cloud requirements, compliance needs, and budget constraints
2. **Pseudocode**: Design infrastructure architecture and service selection
3. **Architecture**: Create Infrastructure as Code templates (Terraform, CloudFormation)
4. **Refinement**: Implement deployment pipelines and monitoring
5. **Completion**: Deploy to production, validate, and optimize

## When to Use This Skill

Trigger this skill when you need to:

- Deploy applications to AWS, GCP, or Azure
- Implement serverless architectures (Lambda, Cloud Functions, Azure Functions)
- Containerize and orchestrate workloads (ECS, GKE, AKS)
- Provision infrastructure with Terraform or CloudFormation
- Design multi-cloud or hybrid cloud architectures
- Optimize cloud costs and performance
- Implement disaster recovery and high availability

## Related Skills

- **[AWS Specialist](aws-specialist/)**: Deep dive into AWS-specific services and best practices
- **[Kubernetes Specialist](kubernetes-specialist/)**: Advanced Kubernetes orchestration across cloud providers
- **terraform-iac**: Infrastructure as Code with Terraform for multi-cloud provisioning
- **docker-containerization**: Container best practices and optimization
- **cicd-intelligent-recovery**: CI/CD pipelines for cloud deployments

## Support and Resources

- **AWS Documentation**: https://docs.aws.amazon.com/
- **GCP Documentation**: https://cloud.google.com/docs
- **Azure Documentation**: https://docs.microsoft.com/azure/
- **Terraform Registry**: https://registry.terraform.io/
- **Cloud Native Computing Foundation**: https://www.cncf.io/

---

**Remember**: Choose the right tool for the job. Not every workload needs to be serverless, and not every application benefits from multi-cloud deployment. Start simple, measure, and iterate.
