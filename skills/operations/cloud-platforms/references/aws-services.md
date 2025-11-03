# AWS Services Reference

Comprehensive guide to Amazon Web Services (AWS) compute, storage, database, networking, and supporting services for cloud-native application development.

## Compute Services

### EC2 (Elastic Compute Cloud)

**Overview**: Virtual machines in the cloud with flexible instance types and pricing models.

**Instance Types**:
- **General Purpose** (T3, T4g, M5, M6i): Balanced CPU, memory, networking
- **Compute Optimized** (C5, C6i, C7g): High-performance processors
- **Memory Optimized** (R5, R6i, X2): Large memory for databases, caching
- **Storage Optimized** (I3, I4i, D3): High sequential read/write to local storage
- **Accelerated Computing** (P4, G5, Inf1): GPU, FPGA, inference acceleration

**Pricing Models**:
- **On-Demand**: Pay per hour/second with no commitment
- **Reserved Instances**: 1-3 year commitment (up to 75% savings)
- **Spot Instances**: Unused capacity (up to 90% savings, interruptible)
- **Savings Plans**: Flexible pricing model based on usage commitment

**Best Practices**:
- Use Auto Scaling Groups for high availability
- Implement lifecycle policies for instance management
- Use EC2 Instance Connect or Session Manager instead of SSH keys
- Enable detailed monitoring for production workloads
- Tag instances for cost allocation and management

**Example: Launch EC2 Instance with Terraform**:
```hcl
resource "aws_instance" "web_server" {
  ami           = "ami-0c55b159cbfafe1f0"  # Amazon Linux 2
  instance_type = "t3.micro"

  vpc_security_group_ids = [aws_security_group.web_sg.id]
  subnet_id              = aws_subnet.public.id

  user_data = <<-EOF
    #!/bin/bash
    yum update -y
    yum install -y httpd
    systemctl start httpd
    systemctl enable httpd
    echo "Hello from EC2!" > /var/www/html/index.html
  EOF

  tags = {
    Name = "web-server-${var.environment}"
  }
}
```

---

### Lambda

**Overview**: Serverless compute service that runs code in response to events without provisioning servers.

**Key Features**:
- **Event-driven execution**: Triggered by S3, DynamoDB, API Gateway, EventBridge, etc.
- **Automatic scaling**: Scales from 0 to 10,000+ concurrent executions
- **Pay-per-use**: Charged per request and compute time (GB-seconds)
- **Integrated monitoring**: CloudWatch Logs and X-Ray tracing built-in

**Supported Runtimes**:
- Node.js 18.x, 16.x
- Python 3.11, 3.10, 3.9
- Java 17, 11, 8
- Go 1.x
- .NET 7, 6
- Ruby 3.2
- Custom runtimes via Lambda Layers

**Limits**:
- **Timeout**: 15 minutes maximum
- **Memory**: 128 MB to 10,240 MB (CPU scales with memory)
- **Package size**: 50 MB zipped, 250 MB unzipped
- **Concurrent executions**: 1,000 per region (soft limit, can increase)

**Best Practices**:
- Keep deployment packages small (use Lambda Layers for dependencies)
- Use environment variables for configuration
- Implement proper error handling and retries
- Enable X-Ray tracing for distributed systems
- Use Provisioned Concurrency for latency-sensitive workloads
- Implement idempotency for event processing

**Example: Lambda Function with API Gateway**:
```javascript
exports.handler = async (event) => {
  console.log('Event:', JSON.stringify(event));

  const { body, httpMethod, path } = event;

  if (httpMethod === 'GET' && path === '/health') {
    return {
      statusCode: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: 'healthy' })
    };
  }

  try {
    const data = JSON.parse(body);

    // Process data
    const result = await processData(data);

    return {
      statusCode: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ result })
    };
  } catch (error) {
    console.error('Error:', error);

    return {
      statusCode: 500,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: error.message })
    };
  }
};
```

---

### ECS & Fargate

**Overview**: Container orchestration services for Docker containers.

**ECS (Elastic Container Service)**:
- Managed container orchestration
- Deep integration with AWS services
- Two launch types: EC2 (manage infrastructure) and Fargate (serverless)

**Fargate**:
- Serverless compute for containers
- No EC2 instances to manage
- Pay for vCPU and memory resources used

**Key Concepts**:
- **Task Definition**: Blueprint for containers (image, CPU, memory, environment)
- **Service**: Maintains desired count of tasks, load balancing, auto-scaling
- **Cluster**: Logical grouping of tasks or services

**Best Practices**:
- Use Fargate for simpler operations and cost-effective scaling
- Use ECS on EC2 for specialized instance types or cost optimization
- Implement health checks for containers
- Use Service Discovery for inter-service communication
- Store secrets in Secrets Manager, not environment variables

**Example: ECS Service with Fargate**:
```hcl
resource "aws_ecs_task_definition" "app" {
  family                   = "app-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "app"
    image = "123456789012.dkr.ecr.us-east-1.amazonaws.com/app:latest"

    portMappings = [{
      containerPort = 8080
      protocol      = "tcp"
    }]

    environment = [
      { name = "NODE_ENV", value = "production" }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/app"
        "awslogs-region"        = "us-east-1"
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

resource "aws_ecs_service" "app" {
  name            = "app-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 3
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = aws_subnet.private[*].id
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "app"
    container_port   = 8080
  }
}
```

---

## Storage Services

### S3 (Simple Storage Service)

**Overview**: Object storage with 99.999999999% (11 nines) durability.

**Storage Classes**:
- **S3 Standard**: Frequently accessed data (millisecond latency)
- **S3 Intelligent-Tiering**: Automatic cost optimization
- **S3 Standard-IA**: Infrequent access (millisecond latency, lower storage cost)
- **S3 One Zone-IA**: Infrequent access in single AZ (20% cheaper than Standard-IA)
- **S3 Glacier Instant Retrieval**: Archive with millisecond retrieval
- **S3 Glacier Flexible Retrieval**: Archive with 1-5 minute retrieval
- **S3 Glacier Deep Archive**: Lowest cost archive (12-hour retrieval)

**Key Features**:
- **Versioning**: Keep multiple versions of objects
- **Lifecycle Policies**: Automate transitions to cheaper storage classes
- **Replication**: Cross-region or same-region replication
- **Event Notifications**: Trigger Lambda, SQS, SNS on object events
- **Object Lock**: WORM (Write Once Read Many) compliance

**Best Practices**:
- Enable versioning for critical data
- Use lifecycle policies to reduce costs
- Enable encryption (SSE-S3, SSE-KMS, or client-side)
- Use S3 Transfer Acceleration for global uploads
- Implement bucket policies for access control
- Enable access logging for auditing

**Example: S3 Bucket with Lifecycle Policy**:
```hcl
resource "aws_s3_bucket" "data" {
  bucket = "my-data-bucket-${var.environment}"
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "archive-old-versions"
    status = "Enabled"

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }
}
```

---

### EBS (Elastic Block Store)

**Overview**: Persistent block storage for EC2 instances.

**Volume Types**:
- **gp3 (General Purpose SSD)**: 3,000-16,000 IOPS, 125-1,000 MB/s (recommended for most workloads)
- **gp2 (General Purpose SSD)**: 100-16,000 IOPS, 3 IOPS per GB
- **io2 (Provisioned IOPS SSD)**: Up to 64,000 IOPS, 99.999% durability
- **io2 Block Express**: Up to 256,000 IOPS, 4,000 MB/s
- **st1 (Throughput Optimized HDD)**: 500 IOPS, 500 MB/s (big data, log processing)
- **sc1 (Cold HDD)**: 250 IOPS, 250 MB/s (infrequently accessed data)

**Best Practices**:
- Use gp3 for most workloads (better price/performance than gp2)
- Use io2 for databases requiring high IOPS
- Enable EBS encryption by default
- Create regular snapshots for backups
- Use EBS Fast Snapshot Restore for quick recovery

---

## Database Services

### RDS (Relational Database Service)

**Overview**: Managed relational database service supporting multiple engines.

**Supported Engines**:
- **PostgreSQL**: Open-source, advanced features
- **MySQL**: Popular open-source database
- **MariaDB**: MySQL fork with additional features
- **Oracle**: Enterprise database
- **SQL Server**: Microsoft database
- **Amazon Aurora**: MySQL/PostgreSQL-compatible, 5x/3x performance

**Key Features**:
- Automated backups (1-35 days retention)
- Point-in-time recovery
- Multi-AZ for high availability
- Read replicas for read scaling
- Automated software patching
- Monitoring with CloudWatch and Performance Insights

**Best Practices**:
- Use Multi-AZ for production databases
- Enable automated backups
- Use read replicas to offload read traffic
- Enable Performance Insights for query analysis
- Use parameter groups for custom configuration
- Implement proper security groups and IAM database authentication

**Example: RDS PostgreSQL with Multi-AZ**:
```hcl
resource "aws_db_instance" "postgres" {
  identifier = "app-db-${var.environment}"

  engine         = "postgres"
  engine_version = "15.3"
  instance_class = "db.t3.medium"

  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = "appdb"
  username = "admin"
  password = random_password.db_password.result

  multi_az               = true
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.database.id]

  backup_retention_period = 30
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "app-db-final-snapshot-${formatdate("YYYY-MM-DD", timestamp())}"

  performance_insights_enabled = true

  tags = {
    Environment = var.environment
  }
}
```

---

### DynamoDB

**Overview**: Fully managed NoSQL database with single-digit millisecond latency.

**Key Features**:
- **Key-value and document data models**
- **On-demand or provisioned capacity modes**
- **Global Tables**: Multi-region replication with active-active writes
- **DynamoDB Streams**: Change data capture for event-driven architectures
- **Point-in-time recovery** (PITR)
- **DAX (DynamoDB Accelerator)**: In-memory caching for microsecond latency

**Capacity Modes**:
- **On-Demand**: Pay per request, automatic scaling
- **Provisioned**: Define read/write capacity units (cheaper for predictable workloads)

**Best Practices**:
- Design partition keys for even distribution
- Use on-demand mode for unpredictable traffic
- Enable point-in-time recovery for critical tables
- Use Global Secondary Indexes (GSI) for query flexibility
- Implement DynamoDB Streams for event processing
- Enable encryption at rest

---

## Networking Services

### VPC (Virtual Private Cloud)

**Overview**: Isolated virtual network in AWS cloud.

**Key Components**:
- **Subnets**: Public (internet-accessible) and private (internal-only)
- **Internet Gateway**: Allow public subnet resources to access internet
- **NAT Gateway**: Allow private subnet resources to access internet
- **Route Tables**: Control traffic routing
- **Security Groups**: Stateful firewall for instances
- **Network ACLs**: Stateless firewall for subnets

**Best Practices**:
- Use multiple Availability Zones for high availability
- Separate public and private subnets
- Use NAT Gateways in each AZ for redundancy
- Implement security groups with least privilege
- Enable VPC Flow Logs for network monitoring

---

### CloudFront

**Overview**: Global content delivery network (CDN) with 450+ edge locations.

**Use Cases**:
- Static website hosting
- API acceleration
- Video streaming
- Software distribution

**Best Practices**:
- Use origin failover for high availability
- Enable compression for faster delivery
- Set appropriate cache TTLs
- Use Lambda@Edge for edge computing
- Implement geo-restrictions for compliance

---

## Cost Optimization

### Reserved Instances & Savings Plans

**Reserved Instances**:
- 1-3 year commitment for EC2, RDS, ElastiCache, Redshift
- Up to 75% savings vs. On-Demand
- Standard (no flexibility) or Convertible (change instance type)

**Savings Plans**:
- Flexible pricing model based on usage commitment ($/hour)
- Compute Savings Plans (EC2, Fargate, Lambda) - up to 66% savings
- EC2 Instance Savings Plans - up to 72% savings

### Right-Sizing

- Use AWS Compute Optimizer for recommendations
- Monitor CloudWatch metrics to identify underutilized resources
- Use T3/T4g burstable instances for variable workloads

### Storage Optimization

- Use S3 Intelligent-Tiering for automatic cost optimization
- Implement lifecycle policies for S3 and EBS snapshots
- Delete unused EBS volumes and snapshots

---

## Related Resources

- [AWS Documentation](https://docs.aws.amazon.com/)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [AWS Pricing Calculator](https://calculator.aws/)
- [AWS Architecture Center](https://aws.amazon.com/architecture/)
