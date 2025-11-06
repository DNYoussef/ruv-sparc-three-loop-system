---
name: aws-specialist
description: AWS cloud specialist for infrastructure as code with CloudFormation/CDK, serverless with Lambda, container orchestration with ECS/Fargate, database management with RDS, storage with S3/CloudFront CDN, and deployment automation. Use when deploying to AWS, optimizing cloud costs, implementing auto-scaling, or requiring AWS-specific best practices. Handles IAM policies, VPC networking, monitoring with CloudWatch, and multi-region deployments.
category: Cloud Platforms
complexity: High
triggers: ["aws", "cloudformation", "lambda", "ecs", "fargate", "rds", "s3", "cloudfront", "aws deployment", "aws cdk", "cloudwatch", "auto-scaling"]
---

# AWS Specialist

Expert AWS cloud infrastructure design, deployment, and optimization for production-grade applications.

## Purpose

Comprehensive AWS expertise across IaC (CloudFormation, CDK), serverless (Lambda), containers (ECS/Fargate), databases (RDS, DynamoDB), storage (S3), CDN (CloudFront), and DevOps automation. Ensures AWS architectures are secure, cost-effective, and scalable.

## When to Use

- Deploying applications to AWS
- Creating infrastructure as code templates
- Setting up serverless architectures
- Configuring auto-scaling and load balancing
- Implementing multi-region deployments
- Optimizing AWS costs (FinOps)
- Securing AWS resources with IAM

## Prerequisites

**Required**: AWS account, AWS CLI installed, basic understanding of cloud concepts

**Agent Assignments**: `cicd-engineer`, `system-architect`, `security-manager`, `perf-analyzer`

## Core Workflows

### Workflow 1: AWS CDK Infrastructure as Code

**Step 1: Initialize CDK Project**

```bash
mkdir my-infra && cd my-infra
npx cdk init app --language typescript
npm install @aws-cdk/aws-lambda @aws-cdk/aws-apigateway @aws-cdk/aws-dynamodb
```

**Step 2: Define Lambda + API Gateway Stack**

```typescript
// lib/api-stack.ts
import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';

export class ApiStack extends cdk.Stack {
  constructor(scope: cdk.App, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // DynamoDB table
    const table = new dynamodb.Table(this, 'ItemsTable', {
      partitionKey: { name: 'id', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY, // ONLY for dev
    });

    // Lambda function
    const handler = new lambda.Function(this, 'ItemsHandler', {
      runtime: lambda.Runtime.NODEJS_18_X,
      code: lambda.Code.fromAsset('lambda'),
      handler: 'index.handler',
      environment: {
        TABLE_NAME: table.tableName,
      },
    });

    table.grantReadWriteData(handler);

    // API Gateway
    const api = new apigateway.RestApi(this, 'ItemsApi', {
      restApiName: 'Items Service',
    });

    const items = api.root.addResource('items');
    items.addMethod('GET', new apigateway.LambdaIntegration(handler));
    items.addMethod('POST', new apigateway.LambdaIntegration(handler));
  }
}
```

**Step 3: Deploy Stack**

```bash
# Bootstrap CDK (first time only)
cdk bootstrap

# Deploy
cdk deploy
```

### Workflow 2: ECS Fargate Deployment

**Step 1: Create Fargate Service with CDK**

```typescript
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';

export class FargateStack extends cdk.Stack {
  constructor(scope: cdk.App, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const vpc = new ec2.Vpc(this, 'MyVpc', { maxAzs: 2 });

    const cluster = new ecs.Cluster(this, 'MyCluster', { vpc });

    const taskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef', {
      memoryLimitMiB: 512,
      cpu: 256,
    });

    taskDefinition.addContainer('web', {
      image: ecs.ContainerImage.fromRegistry('nginx'),
      portMappings: [{ containerPort: 80 }],
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: 'MyApp' }),
    });

    const service = new ecs.FargateService(this, 'Service', {
      cluster,
      taskDefinition,
      desiredCount: 2,
    });

    const lb = new elbv2.ApplicationLoadBalancer(this, 'LB', {
      vpc,
      internetFacing: true,
    });

    const listener = lb.addListener('Listener', { port: 80 });
    listener.addTargets('ECS', {
      port: 80,
      targets: [service],
      healthCheck: { path: '/' },
    });
  }
}
```

### Workflow 3: RDS Database with Backups

```typescript
import * as rds from 'aws-cdk-lib/aws-rds';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';

const dbSecret = new secretsmanager.Secret(this, 'DBSecret', {
  generateSecretString: {
    secretStringTemplate: JSON.stringify({ username: 'admin' }),
    generateStringKey: 'password',
    excludePunctuation: true,
  },
});

const db = new rds.DatabaseInstance(this, 'MyDatabase', {
  engine: rds.DatabaseInstanceEngine.postgres({
    version: rds.PostgresEngineVersion.VER_15_3,
  }),
  instanceType: ec2.InstanceType.of(
    ec2.InstanceClass.T3,
    ec2.InstanceSize.MICRO
  ),
  vpc,
  credentials: rds.Credentials.fromSecret(dbSecret),
  backupRetention: cdk.Duration.days(7),
  deleteAutomatedBackups: false,
  removalPolicy: cdk.RemovalPolicy.SNAPSHOT,
});
```

### Workflow 4: S3 + CloudFront CDN

```typescript
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';

const bucket = new s3.Bucket(this, 'WebsiteBucket', {
  publicReadAccess: false,
  blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});

const distribution = new cloudfront.Distribution(this, 'Distribution', {
  defaultBehavior: {
    origin: new origins.S3Origin(bucket),
    viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
  },
  defaultRootObject: 'index.html',
});
```

## Best Practices

**1. Use IAM Least Privilege**
```typescript
// ✅ GOOD: Specific permissions
lambda.addToRolePolicy(new iam.PolicyStatement({
  actions: ['dynamodb:GetItem', 'dynamodb:PutItem'],
  resources: [table.tableArn],
}));

// ❌ BAD: Overly permissive
lambda.addToRolePolicy(new iam.PolicyStatement({
  actions: ['*'],
  resources: ['*'],
}));
```

**2. Enable Encryption**
```typescript
const bucket = new s3.Bucket(this, 'Bucket', {
  encryption: s3.BucketEncryption.S3_MANAGED, // or KMS
});
```

**3. Use Secrets Manager for Credentials**
```typescript
// ✅ GOOD: Secrets Manager
const secret = secretsmanager.Secret.fromSecretNameV2(this, 'Secret', 'my-secret');

// ❌ BAD: Hardcoded in environment
environment: { API_KEY: 'hardcoded-key-123' }
```

**4. Enable Auto-Scaling**
```typescript
const scaling = service.autoScaleTaskCount({ maxCapacity: 10 });
scaling.scaleOnCpuUtilization('CpuScaling', {
  targetUtilizationPercent: 70,
});
```

**5. Cost Optimization**
- Use Spot Instances for non-critical workloads
- Enable S3 Intelligent-Tiering
- Use Lambda for short-lived tasks
- Set up CloudWatch billing alarms

## Quality Criteria

- ✅ Infrastructure defined in code (CDK/CloudFormation)
- ✅ All secrets in Secrets Manager
- ✅ Encryption at rest and in transit
- ✅ Auto-scaling configured
- ✅ CloudWatch monitoring enabled
- ✅ Cost allocation tags applied
- ✅ Multi-AZ for production databases

## Troubleshooting

**Issue**: CDK deployment fails with "not authorized"
**Solution**: Check AWS credentials (`aws sts get-caller-identity`), ensure IAM permissions

**Issue**: Lambda timeout errors
**Solution**: Increase timeout (max 15 minutes), check VPC NAT gateway if Lambda in VPC

**Issue**: High RDS costs
**Solution**: Use Aurora Serverless v2 for variable workloads, stop dev databases overnight

## Related Skills

- `terraform-iac`: Multi-cloud IaC
- `docker-containerization`: Container best practices
- `kubernetes-specialist`: K8s on EKS
- `opentelemetry-observability`: Distributed tracing

## Tools

- AWS CDK: Infrastructure as code in TypeScript/Python
- AWS CLI: Command-line interface
- CloudFormation: Declarative IaC
- Copilot CLI: ECS simplified deployment

## MCP Tools

- `mcp__flow-nexus__sandbox_execute` for AWS CLI commands
- `mcp__memory-mcp__memory_store` for AWS architecture patterns

## Success Metrics

- Deployment time: <15 minutes for full stack
- Infrastructure drift: 0%
- Cost optimization: 20-30% reduction with reserved instances
- Availability: 99.9%+ with multi-AZ

---

**Skill Version**: 1.0.0
**Last Updated**: 2025-11-02
