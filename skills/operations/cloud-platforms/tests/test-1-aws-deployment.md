# Test 1: AWS Lambda and ECS Fargate Deployment

## Test Objective
Validate AWS deployment automation for Lambda functions and ECS Fargate services using the deployment scripts and Terraform templates.

## Prerequisites
- AWS CLI configured with valid credentials
- AWS account with sufficient permissions
- Docker installed for container builds
- Terraform >= 1.0 installed
- Python 3.11+ installed

## Test Scenarios

### Scenario 1.1: Lambda Function Deployment

**Setup:**
```bash
# Create test Lambda function
mkdir -p test-lambda
cd test-lambda

cat > index.py << 'EOF'
import json

def handler(event, context):
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Hello from Lambda!',
            'event': event
        })
    }
EOF

# Package function
zip function.zip index.py

# Set Lambda execution role
export AWS_LAMBDA_ROLE="arn:aws:iam::ACCOUNT_ID:role/lambda-execution-role"
```

**Test Execution:**
```bash
# Deploy Lambda function
python ../resources/scripts/deploy_aws.py \
  --region us-east-1 \
  lambda \
  --name test-function \
  --zip function.zip \
  --handler index.handler \
  --runtime python3.11 \
  --memory 256 \
  --timeout 30

# Verify deployment
aws lambda get-function --function-name test-function --region us-east-1

# Invoke function
aws lambda invoke \
  --function-name test-function \
  --region us-east-1 \
  --payload '{"test": "data"}' \
  response.json

cat response.json
```

**Expected Results:**
- Function deploys successfully
- Function returns 200 status code
- Response contains expected message
- Execution time < 30 seconds
- Memory usage < 256 MB

**Success Criteria:**
- ✅ Lambda function created or updated
- ✅ Function invocation successful
- ✅ Correct response payload returned
- ✅ No errors in CloudWatch Logs

---

### Scenario 1.2: ECS Fargate Service Deployment

**Setup:**
```bash
# Create ECS task definition
cat > task-definition.json << 'EOF'
{
  "family": "test-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [
    {
      "name": "nginx",
      "image": "nginx:alpine",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 80,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/test-task",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
EOF

# Create log group
aws logs create-log-group --log-group-name /ecs/test-task --region us-east-1
```

**Test Execution:**
```bash
# Deploy ECS service
python ../resources/scripts/deploy_aws.py \
  --region us-east-1 \
  ecs \
  --cluster test-cluster \
  --service test-service \
  --task-def task-definition.json \
  --count 2

# Verify deployment
aws ecs describe-services \
  --cluster test-cluster \
  --services test-service \
  --region us-east-1

# Check task status
aws ecs list-tasks \
  --cluster test-cluster \
  --service-name test-service \
  --region us-east-1
```

**Expected Results:**
- ECS service created successfully
- 2 tasks running in Fargate
- Tasks pass health checks
- Service is stable
- No deployment errors

**Success Criteria:**
- ✅ ECS service created or updated
- ✅ Desired count matches actual running tasks
- ✅ All tasks in RUNNING state
- ✅ No STOPPED tasks with errors

---

### Scenario 1.3: Terraform AWS Infrastructure

**Setup:**
```bash
# Create test directory
mkdir -p terraform-test
cd terraform-test
cp ../resources/templates/aws-infra.tf main.tf

# Create variables file
cat > terraform.tfvars << 'EOF'
project_name     = "test-app"
environment      = "dev"
region           = "us-east-1"
container_image  = "nginx:alpine"
container_port   = 80
desired_count    = 2
cpu              = 256
memory           = 512
enable_rds       = false
EOF
```

**Test Execution:**
```bash
# Initialize Terraform
terraform init

# Validate configuration
terraform validate

# Format check
terraform fmt -check

# Create plan
terraform plan -out=tfplan

# Apply infrastructure
terraform apply tfplan

# Get outputs
terraform output -json > outputs.json
cat outputs.json

# Test ALB endpoint
ALB_DNS=$(terraform output -raw alb_dns_name)
curl -I http://$ALB_DNS

# Wait for ECS service to stabilize
aws ecs wait services-stable \
  --cluster $(terraform output -raw ecs_cluster_name) \
  --services $(terraform output -raw ecs_service_name) \
  --region us-east-1
```

**Expected Results:**
- All Terraform resources created successfully
- VPC with public/private subnets
- ALB with target group and listener
- ECS cluster with Fargate service
- 2 running ECS tasks
- ALB health checks passing
- HTTP request to ALB returns 200

**Success Criteria:**
- ✅ `terraform apply` succeeds with no errors
- ✅ All outputs populated correctly
- ✅ ALB accessible via DNS name
- ✅ ECS service stable with 2/2 tasks running
- ✅ Health checks passing

---

### Scenario 1.4: CloudFormation Stack Deployment

**Setup:**
```bash
# Create CloudFormation template
cat > stack.yaml << 'EOF'
AWSTemplateFormatVersion: '2010-09-09'
Description: Test Lambda function with API Gateway

Resources:
  TestFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: cfn-test-function
      Runtime: python3.11
      Handler: index.handler
      Role: !GetAtt LambdaExecutionRole.Arn
      Code:
        ZipFile: |
          import json
          def handler(event, context):
              return {
                  'statusCode': 200,
                  'body': json.dumps({'message': 'Hello from CloudFormation!'})
              }

  LambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

Outputs:
  FunctionArn:
    Value: !GetAtt TestFunction.Arn
    Description: Lambda Function ARN
EOF
```

**Test Execution:**
```bash
# Deploy CloudFormation stack
python ../resources/scripts/deploy_aws.py \
  --region us-east-1 \
  cfn \
  --stack test-stack \
  --template stack.yaml

# Verify stack status
aws cloudformation describe-stacks \
  --stack-name test-stack \
  --region us-east-1 \
  --query 'Stacks[0].StackStatus'

# Get outputs
aws cloudformation describe-stacks \
  --stack-name test-stack \
  --region us-east-1 \
  --query 'Stacks[0].Outputs'

# Test Lambda function
aws lambda invoke \
  --function-name cfn-test-function \
  --region us-east-1 \
  response.json

cat response.json
```

**Expected Results:**
- CloudFormation stack created successfully
- Lambda function deployed
- IAM role created with proper permissions
- Function invocation successful
- Stack outputs available

**Success Criteria:**
- ✅ Stack status is `CREATE_COMPLETE` or `UPDATE_COMPLETE`
- ✅ All resources created successfully
- ✅ Function invocation returns 200
- ✅ Stack outputs correctly populated

---

## Cleanup

```bash
# Scenario 1.1 cleanup
aws lambda delete-function --function-name test-function --region us-east-1

# Scenario 1.2 cleanup
aws ecs delete-service --cluster test-cluster --service test-service --force --region us-east-1
aws ecs delete-cluster --cluster test-cluster --region us-east-1
aws logs delete-log-group --log-group-name /ecs/test-task --region us-east-1

# Scenario 1.3 cleanup
cd terraform-test
terraform destroy -auto-approve

# Scenario 1.4 cleanup
aws cloudformation delete-stack --stack-name test-stack --region us-east-1
```

## Test Report Template

```markdown
# AWS Deployment Test Report

**Date:** YYYY-MM-DD
**Tester:** Name
**Environment:** AWS Account ID

## Test Results Summary

| Scenario | Status | Duration | Notes |
|----------|--------|----------|-------|
| 1.1 Lambda | ✅/❌ | XXs | |
| 1.2 ECS Fargate | ✅/❌ | XXXs | |
| 1.3 Terraform | ✅/❌ | XXXs | |
| 1.4 CloudFormation | ✅/❌ | XXXs | |

## Issues Encountered

1. **Issue:** Description
   - **Resolution:** How it was fixed
   - **Impact:** Severity level

## Recommendations

- Improvement suggestions
- Best practices identified
- Script enhancements needed

## Sign-off

- [ ] All tests passed
- [ ] Cleanup completed
- [ ] Documentation updated
```

## Notes

- Ensure AWS credentials have appropriate permissions
- Lambda execution role ARN must be set before Lambda deployment
- ECS cluster must exist before service deployment
- Monitor CloudWatch Logs for debugging
- Clean up resources to avoid unnecessary charges
- Test in non-production environment first
