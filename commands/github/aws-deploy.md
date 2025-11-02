# AWS Deployment Command

**Category**: External Integrations
**Phase**: 4 - Deployment & Integration
**Complexity**: High

## Purpose

Automated deployment to AWS infrastructure with infrastructure-as-code validation, multi-region support, and rollback capabilities.

## Usage

```bash
/aws-deploy [options]

Options:
  --service <name>        AWS service (ec2|ecs|lambda|s3|cloudfront)
  --region <region>       AWS region (default: us-east-1)
  --env <environment>     Environment (dev|staging|prod)
  --profile <name>        AWS CLI profile
  --stack <name>          CloudFormation stack name
  --rollback-on-failure   Auto-rollback on deployment failure
  --dry-run               Preview deployment without execution
```

## Examples

```bash
# Deploy Lambda function
/aws-deploy --service lambda --region us-west-2 --env prod --profile production

# Deploy ECS container with auto-rollback
/aws-deploy --service ecs --region eu-west-1 --env staging --rollback-on-failure

# Deploy static site to S3 + CloudFront
/aws-deploy --service s3 --region us-east-1 --env prod --stack my-website

# Dry-run deployment preview
/aws-deploy --service ec2 --region ap-south-1 --env dev --dry-run
```

## Workflow

1. **Pre-Deployment Validation**
   - Verify AWS credentials and permissions
   - Validate CloudFormation/SAM templates
   - Check resource quotas and limits
   - Run security compliance checks (AWS Config)

2. **Infrastructure Provisioning**
   - Create/update CloudFormation stacks
   - Configure Auto Scaling groups
   - Set up load balancers and target groups
   - Configure VPC, subnets, security groups

3. **Application Deployment**
   - Build and package application artifacts
   - Upload to S3 or ECR registry
   - Deploy to Lambda/ECS/EC2
   - Update Route53 DNS records

4. **Post-Deployment**
   - Run smoke tests on deployed resources
   - Configure CloudWatch alarms and dashboards
   - Enable X-Ray tracing
   - Generate deployment report

5. **Rollback on Failure**
   - Monitor deployment health checks
   - Auto-rollback to previous stable version
   - Restore database snapshots if needed
   - Notify team via SNS/Slack

## Integration Points

- **AWS CLI**: Infrastructure management
- **AWS SDK**: Programmatic resource control
- **CloudFormation**: Infrastructure as code
- **SAM/CDK**: Serverless deployments
- **CodeDeploy**: Blue/green deployments
- **Systems Manager**: Parameter Store integration

## Security

- ✅ Use IAM roles with least privilege
- ✅ Encrypt data at rest (KMS)
- ✅ Encrypt data in transit (TLS 1.2+)
- ✅ Enable CloudTrail logging
- ✅ Use AWS Secrets Manager for sensitive data
- ✅ Implement VPC security groups

## Prerequisites

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure --profile production
# Enter: Access Key ID, Secret Access Key, Region, Output format

# Verify access
aws sts get-caller-identity --profile production
```

## Environment Variables

```bash
AWS_PROFILE=production
AWS_REGION=us-east-1
AWS_DEFAULT_OUTPUT=json
AWS_PAGER=""
```

## Command Output

```json
{
  "deployment_id": "deploy-abc123",
  "service": "lambda",
  "region": "us-west-2",
  "environment": "prod",
  "status": "SUCCESS",
  "resources_created": [
    "my-function-prod",
    "my-function-role",
    "my-function-log-group"
  ],
  "endpoint": "https://abc123.execute-api.us-west-2.amazonaws.com/prod",
  "deployment_time": "2m 34s",
  "cost_estimate": "$0.12/month"
}
```

## Error Handling

- **Credential Errors**: Verify AWS_PROFILE and credentials
- **Permission Denied**: Check IAM policies
- **Resource Limits**: Request quota increase
- **Template Validation**: Fix CloudFormation syntax
- **Deployment Failure**: Review CloudWatch logs

## Best Practices

1. Use separate AWS accounts for dev/staging/prod
2. Tag all resources for cost tracking
3. Enable CloudTrail in all regions
4. Use AWS Organizations for multi-account management
5. Implement least privilege IAM policies
6. Enable GuardDuty for threat detection
7. Use AWS Config for compliance monitoring
8. Backup critical resources with AWS Backup

## Related Commands

- `/terraform-apply` - Terraform infrastructure deployment
- `/docker-compose` - Container orchestration
- `/ansible-deploy` - Configuration management
- `/github-actions` - CI/CD automation

## Agent Integration

```javascript
// Researcher: Find AWS best practices
mcp__ruv-swarm__agent_spawn({ type: "researcher", capabilities: ["aws-architecture"] })

// Coder: Implement deployment scripts
mcp__ruv-swarm__agent_spawn({ type: "coder", capabilities: ["aws-sdk", "cloudformation"] })

// DevOps: Configure infrastructure
mcp__ruv-swarm__agent_spawn({ type: "optimizer", capabilities: ["aws-devops"] })
```

## Monitoring

```bash
# CloudWatch Logs
aws logs tail /aws/lambda/my-function --follow --profile production

# CloudWatch Metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Invocations \
  --dimensions Name=FunctionName,Value=my-function \
  --start-time 2025-11-01T00:00:00Z \
  --end-time 2025-11-01T23:59:59Z \
  --period 3600 \
  --statistics Sum

# X-Ray tracing
aws xray get-trace-summaries \
  --start-time 2025-11-01T00:00:00Z \
  --end-time 2025-11-01T23:59:59Z
```

---

**Implementation Status**: Production-Ready
**Last Updated**: 2025-11-01
**Maintainer**: Integration Specialist
