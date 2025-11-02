# Example 1: AWS Lambda Serverless REST API

Build a production-ready serverless REST API using AWS Lambda, API Gateway, and DynamoDB with Infrastructure as Code.

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                   AWS Serverless API                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Internet                                                  │
│     │                                                      │
│     ▼                                                      │
│  ┌──────────────────┐                                     │
│  │  Amazon CloudFront                                      │
│  │  (CDN + SSL/TLS) │                                     │
│  └────────┬─────────┘                                     │
│           │                                                │
│           ▼                                                │
│  ┌──────────────────┐                                     │
│  │  API Gateway     │                                     │
│  │  (REST API)      │                                     │
│  └────────┬─────────┘                                     │
│           │                                                │
│           ▼                                                │
│  ┌──────────────────┐     ┌──────────────────┐           │
│  │  Lambda Function │────▶│  DynamoDB Table  │           │
│  │  (Node.js/Python)│     │  (NoSQL)         │           │
│  └──────────────────┘     └──────────────────┘           │
│           │                                                │
│           ▼                                                │
│  ┌──────────────────┐                                     │
│  │  CloudWatch Logs │                                     │
│  │  (Monitoring)    │                                     │
│  └──────────────────┘                                     │
└────────────────────────────────────────────────────────────┘
```

## Use Case

RESTful API for a task management application with the following endpoints:

- `POST /tasks` - Create a new task
- `GET /tasks` - List all tasks
- `GET /tasks/{id}` - Get a specific task
- `PUT /tasks/{id}` - Update a task
- `DELETE /tasks/{id}` - Delete a task

## Prerequisites

```bash
# Install AWS CLI and configure credentials
aws configure

# Install AWS SAM CLI for local testing
pip install aws-sam-cli

# Install Terraform
brew install terraform  # macOS
# or download from https://www.terraform.io/downloads
```

## Project Structure

```
serverless-api/
├── terraform/
│   ├── main.tf              # Main Terraform configuration
│   ├── variables.tf         # Input variables
│   ├── outputs.tf           # Output values
│   └── lambda.tf            # Lambda-specific resources
├── src/
│   ├── handlers/
│   │   ├── createTask.js    # POST /tasks handler
│   │   ├── getTasks.js      # GET /tasks handler
│   │   ├── getTask.js       # GET /tasks/{id} handler
│   │   ├── updateTask.js    # PUT /tasks/{id} handler
│   │   └── deleteTask.js    # DELETE /tasks/{id} handler
│   ├── lib/
│   │   ├── dynamodb.js      # DynamoDB helper functions
│   │   └── validation.js    # Input validation
│   └── package.json
├── tests/
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
└── README.md
```

## Implementation

### Step 1: Terraform Infrastructure

**terraform/main.tf**:
```hcl
terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "my-terraform-state"
    key    = "serverless-api/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# DynamoDB Table
resource "aws_dynamodb_table" "tasks" {
  name           = "${var.project_name}-tasks-${var.environment}"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "taskId"

  attribute {
    name = "taskId"
    type = "S"
  }

  attribute {
    name = "createdAt"
    type = "N"
  }

  global_secondary_index {
    name            = "CreatedAtIndex"
    hash_key        = "createdAt"
    projection_type = "ALL"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled = true
  }

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# API Gateway
resource "aws_apigatewayv2_api" "tasks_api" {
  name          = "${var.project_name}-api-${var.environment}"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = var.cors_origins
    allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers = ["Content-Type", "Authorization"]
    max_age       = 3600
  }
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.tasks_api.id
  name        = var.environment
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_logs.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }
}

# CloudWatch Log Group for API Gateway
resource "aws_cloudwatch_log_group" "api_logs" {
  name              = "/aws/apigateway/${var.project_name}-${var.environment}"
  retention_in_days = 30
}

# Lambda IAM Role
resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_dynamodb" {
  name = "dynamodb-access"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:Query",
        "dynamodb:Scan"
      ]
      Resource = [
        aws_dynamodb_table.tasks.arn,
        "${aws_dynamodb_table.tasks.arn}/index/*"
      ]
    }]
  })
}
```

**terraform/lambda.tf**:
```hcl
# Lambda Layer for shared dependencies
resource "aws_lambda_layer_version" "dependencies" {
  filename            = "lambda_layer.zip"
  layer_name          = "${var.project_name}-dependencies-${var.environment}"
  compatible_runtimes = ["nodejs18.x"]
}

# Lambda Function for Create Task
resource "aws_lambda_function" "create_task" {
  filename         = "create_task.zip"
  function_name    = "${var.project_name}-create-task-${var.environment}"
  role            = aws_iam_role.lambda_role.arn
  handler         = "handlers/createTask.handler"
  runtime         = "nodejs18.x"
  timeout         = 30
  memory_size     = 256

  layers = [aws_lambda_layer_version.dependencies.arn]

  environment {
    variables = {
      TABLE_NAME  = aws_dynamodb_table.tasks.name
      ENVIRONMENT = var.environment
    }
  }

  tracing_config {
    mode = "Active"
  }
}

resource "aws_cloudwatch_log_group" "create_task_logs" {
  name              = "/aws/lambda/${aws_lambda_function.create_task.function_name}"
  retention_in_days = 30
}

# API Gateway Integration
resource "aws_apigatewayv2_integration" "create_task" {
  api_id           = aws_apigatewayv2_api.tasks_api.id
  integration_type = "AWS_PROXY"

  integration_uri    = aws_lambda_function.create_task.invoke_arn
  integration_method = "POST"
}

resource "aws_apigatewayv2_route" "create_task" {
  api_id    = aws_apigatewayv2_api.tasks_api.id
  route_key = "POST /tasks"
  target    = "integrations/${aws_apigatewayv2_integration.create_task.id}"
}

resource "aws_lambda_permission" "create_task_apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.create_task.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.tasks_api.execution_arn}/*/*"
}

# Repeat similar resources for other Lambda functions:
# - get_tasks (GET /tasks)
# - get_task (GET /tasks/{id})
# - update_task (PUT /tasks/{id})
# - delete_task (DELETE /tasks/{id})
```

### Step 2: Lambda Function Code

**src/handlers/createTask.js**:
```javascript
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, PutCommand } = require('@aws-sdk/lib-dynamodb');
const { v4: uuidv4 } = require('uuid');
const { validateTask } = require('../lib/validation');

const client = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(client);

exports.handler = async (event) => {
  try {
    // Parse request body
    const body = JSON.parse(event.body);

    // Validate input
    const validation = validateTask(body);
    if (!validation.valid) {
      return {
        statusCode: 400,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        },
        body: JSON.stringify({
          error: 'Validation failed',
          details: validation.errors
        })
      };
    }

    // Create task object
    const task = {
      taskId: uuidv4(),
      title: body.title,
      description: body.description || '',
      status: 'pending',
      priority: body.priority || 'medium',
      createdAt: Date.now(),
      updatedAt: Date.now()
    };

    // Save to DynamoDB
    await docClient.send(new PutCommand({
      TableName: process.env.TABLE_NAME,
      Item: task,
      ConditionExpression: 'attribute_not_exists(taskId)'
    }));

    return {
      statusCode: 201,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      },
      body: JSON.stringify({
        message: 'Task created successfully',
        task
      })
    };

  } catch (error) {
    console.error('Error creating task:', error);

    return {
      statusCode: 500,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      },
      body: JSON.stringify({
        error: 'Internal server error',
        message: error.message
      })
    };
  }
};
```

**src/lib/validation.js**:
```javascript
exports.validateTask = (task) => {
  const errors = [];

  if (!task.title || typeof task.title !== 'string' || task.title.trim().length === 0) {
    errors.push('Title is required and must be a non-empty string');
  }

  if (task.title && task.title.length > 200) {
    errors.push('Title must be 200 characters or less');
  }

  if (task.description && task.description.length > 1000) {
    errors.push('Description must be 1000 characters or less');
  }

  if (task.priority && !['low', 'medium', 'high'].includes(task.priority)) {
    errors.push('Priority must be one of: low, medium, high');
  }

  return {
    valid: errors.length === 0,
    errors
  };
};
```

### Step 3: Deployment

```bash
# Initialize Terraform
cd terraform
terraform init

# Create workspace for environment
terraform workspace new dev

# Plan deployment
terraform plan -var-file="dev.tfvars"

# Deploy infrastructure
terraform apply -var-file="dev.tfvars"

# Package Lambda functions
cd ../src
npm install
npm run package  # Creates .zip files for each Lambda function

# Upload Lambda code
aws lambda update-function-code \
  --function-name tasks-api-create-task-dev \
  --zip-file fileb://dist/create_task.zip
```

### Step 4: Testing

**Local Testing with SAM**:
```bash
# Start local API Gateway and Lambda
sam local start-api

# Test endpoint
curl -X POST http://localhost:3000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Complete project documentation",
    "description": "Write comprehensive README and API docs",
    "priority": "high"
  }'
```

**Integration Testing**:
```bash
# Run integration tests against deployed API
npm test

# Load testing with Artillery
artillery run load-test.yml
```

## Monitoring and Observability

### CloudWatch Metrics

Key metrics to monitor:
- Lambda invocations, errors, duration, throttles
- API Gateway 4xx/5xx errors, latency, request count
- DynamoDB consumed read/write capacity, throttled requests

### CloudWatch Alarms

```hcl
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${var.project_name}-lambda-errors-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 5

  dimensions = {
    FunctionName = aws_lambda_function.create_task.function_name
  }

  alarm_actions = [aws_sns_topic.alerts.arn]
}
```

### X-Ray Tracing

Enable X-Ray for distributed tracing:
```javascript
const AWSXRay = require('aws-xray-sdk-core');
const AWS = AWSXRay.captureAWS(require('aws-sdk'));
```

## Cost Optimization

### Estimated Monthly Cost (Low-Medium Traffic)

| Service | Usage | Cost |
|---------|-------|------|
| Lambda | 1M requests, 256MB, 200ms avg | $0.40 |
| API Gateway | 1M requests | $3.50 |
| DynamoDB | 1M writes, 2M reads | $1.50 |
| CloudWatch Logs | 5GB ingestion, 1GB storage | $2.65 |
| **Total** | | **~$8.05/month** |

### Cost-Saving Tips

1. **Use Lambda Provisioned Concurrency only for critical endpoints**
2. **Enable DynamoDB auto-scaling** for variable workloads
3. **Set CloudWatch log retention** to 30 days (default is indefinite)
4. **Use API Gateway caching** for read-heavy endpoints
5. **Implement Lambda function optimization** (reduce package size, reuse connections)

## Security Best Practices

1. **API Authentication**: Add Lambda authorizer for JWT validation
2. **Encryption**: Enable encryption at rest for DynamoDB
3. **VPC**: Deploy Lambda in VPC if accessing private resources
4. **Secrets Management**: Use AWS Secrets Manager for sensitive data
5. **Input Validation**: Validate all input at API Gateway and Lambda layers
6. **Rate Limiting**: Configure API Gateway throttling
7. **WAF**: Add AWS WAF for protection against common attacks

## Production Readiness Checklist

- [ ] Multi-region deployment for disaster recovery
- [ ] Automated backup and restore for DynamoDB
- [ ] Comprehensive monitoring and alerting
- [ ] Load testing with realistic traffic patterns
- [ ] Security scanning with tools like Snyk, AWS Inspector
- [ ] Cost optimization review
- [ ] Documentation for operations team
- [ ] Runbook for incident response

## Related Resources

- [AWS Lambda Best Practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)
- [DynamoDB Best Practices](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/best-practices.html)
- [API Gateway Throttling](https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-request-throttling.html)
