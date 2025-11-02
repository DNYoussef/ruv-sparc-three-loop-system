---
name: cloud-platforms
description: Multi-cloud deployment and infrastructure management across AWS, GCP, and Azure. Use when deploying applications to cloud platforms, implementing serverless architectures, or managing cloud infrastructure as code. Supports containers, serverless, and traditional compute.
---

# Cloud Platforms - Multi-Cloud Infrastructure

Comprehensive cloud deployment and management for AWS, Google Cloud, and Azure platforms.

## When to Use This Skill

Use when deploying applications to cloud platforms, implementing serverless architectures (Lambda, Cloud Functions), managing containerized workloads (ECS, GKE, AKS), or provisioning cloud infrastructure with Terraform/CloudFormation.

## Supported Platforms

### AWS (Amazon Web Services)
- **Compute**: EC2, Lambda, ECS, Fargate, Batch
- **Storage**: S3, EBS, EFS, Glacier
- **Database**: RDS, DynamoDB, Aurora, Redshift
- **Networking**: VPC, Route 53, CloudFront, API Gateway
- **IaC**: CloudFormation, AWS CDK

### Google Cloud Platform
- **Compute**: Compute Engine, Cloud Functions, GKE, Cloud Run
- **Storage**: Cloud Storage, Persistent Disk, Filestore
- **Database**: Cloud SQL, Firestore, BigQuery, Spanner
- **Networking**: VPC, Cloud CDN, Cloud Load Balancing
- **IaC**: Deployment Manager, Terraform

### Microsoft Azure
- **Compute**: VMs, Azure Functions, AKS, Container Instances
- **Storage**: Blob Storage, Disk Storage, Azure Files
- **Database**: SQL Database, Cosmos DB, Synapse Analytics
- **Networking**: Virtual Network, Traffic Manager, Front Door
- **IaC**: ARM Templates, Bicep, Terraform

## Process

1. **Define requirements**
   - Determine workload type (compute, storage, database)
   - Assess scaling needs
   - Identify compliance requirements
   - Estimate costs

2. **Select platform and services**
   - Choose cloud provider (AWS/GCP/Azure)
   - Pick appropriate services for workload
   - Design for high availability
   - Plan disaster recovery

3. **Provision infrastructure**
   - Use Infrastructure as Code (Terraform, CloudFormation)
   - Implement security best practices
   - Configure networking and access
   - Set up monitoring and logging

4. **Deploy applications**
   - Containerize with Docker
   - Use CI/CD pipelines
   - Implement blue-green or canary deployments
   - Configure auto-scaling

5. **Monitor and optimize**
   - Track resource utilization
   - Optimize costs (right-sizing, spot instances)
   - Review security posture
   - Implement performance improvements

## Best Practices

- **Multi-region**: Deploy across regions for high availability
- **Infrastructure as Code**: Never provision manually
- **Cost Optimization**: Use reserved instances, spot instances
- **Security**: Least privilege IAM, encryption at rest/transit
- **Monitoring**: CloudWatch, Stackdriver, Azure Monitor
