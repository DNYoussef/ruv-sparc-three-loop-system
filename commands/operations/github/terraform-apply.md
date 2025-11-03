# Terraform Infrastructure Command

**Category**: External Integrations
**Phase**: 4 - Deployment & Integration
**Complexity**: High

## Purpose

Infrastructure as Code (IaC) deployment using Terraform with state management, drift detection, and multi-cloud support.

## Usage

```bash
/terraform-apply [action] [options]

Actions:
  init                Initialize Terraform
  plan                Preview changes
  apply               Apply changes
  destroy             Destroy infrastructure
  validate            Validate configuration
  state               Manage state
  import              Import existing resources
  drift-detect        Detect configuration drift

Options:
  --workspace <name>  Terraform workspace (dev|staging|prod)
  --target <resource> Target specific resource
  --var-file <path>   Variable file path
  --auto-approve      Skip confirmation
  --destroy           Destroy infrastructure
  --backend <type>    Backend type (s3|gcs|azurerm|local)
```

## Examples

```bash
# Initialize Terraform with remote backend
/terraform-apply init --backend s3 --workspace prod

# Preview infrastructure changes
/terraform-apply plan --workspace staging --var-file vars/staging.tfvars

# Apply changes with auto-approval
/terraform-apply apply --workspace prod --auto-approve

# Target specific resource
/terraform-apply apply --target aws_instance.web --workspace dev

# Destroy all infrastructure
/terraform-apply destroy --workspace dev --auto-approve

# Detect configuration drift
/terraform-apply drift-detect --workspace prod

# Import existing AWS EC2 instance
/terraform-apply import --resource aws_instance.web --id i-1234567890abcdef0

# Validate configuration
/terraform-apply validate
```

## Terraform Configuration Structure

```
terraform/
├── main.tf              # Main configuration
├── variables.tf         # Input variables
├── outputs.tf           # Output values
├── providers.tf         # Provider configuration
├── backend.tf           # Backend configuration
├── versions.tf          # Terraform/provider versions
├── modules/             # Reusable modules
│   ├── vpc/
│   ├── ec2/
│   └── rds/
├── environments/
│   ├── dev.tfvars
│   ├── staging.tfvars
│   └── prod.tfvars
└── .terraform/          # Terraform state (gitignored)
```

## AWS Multi-Tier Infrastructure Example

### 1. Provider Configuration

```hcl
# providers.tf
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "mycompany-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      ManagedBy   = "Terraform"
      Project     = var.project_name
    }
  }
}
```

### 2. VPC and Network

```hcl
# main.tf - VPC
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "${var.project_name}-vpc"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}
```

### 3. Security Groups

```hcl
# Security group for web servers
resource "aws_security_group" "web" {
  name        = "${var.project_name}-web-sg"
  description = "Security group for web servers"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Security group for database
resource "aws_security_group" "database" {
  name        = "${var.project_name}-db-sg"
  description = "Security group for RDS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.web.id]
  }
}
```

### 4. EC2 Auto Scaling

```hcl
# Launch template
resource "aws_launch_template" "web" {
  name_prefix   = "${var.project_name}-web-"
  image_id      = data.aws_ami.amazon_linux_2.id
  instance_type = var.instance_type

  vpc_security_group_ids = [aws_security_group.web.id]

  user_data = base64encode(templatefile("${path.module}/user-data.sh", {
    db_endpoint = aws_db_instance.main.endpoint
  }))

  iam_instance_profile {
    name = aws_iam_instance_profile.web.name
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.project_name}-web"
    }
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "web" {
  name                = "${var.project_name}-web-asg"
  vpc_zone_identifier = module.vpc.private_subnets
  target_group_arns   = [aws_lb_target_group.web.arn]
  health_check_type   = "ELB"

  min_size         = var.asg_min_size
  max_size         = var.asg_max_size
  desired_capacity = var.asg_desired_capacity

  launch_template {
    id      = aws_launch_template.web.id
    version = "$Latest"
  }
}
```

### 5. Application Load Balancer

```hcl
resource "aws_lb" "web" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.web.id]
  subnets            = module.vpc.public_subnets
}

resource "aws_lb_target_group" "web" {
  name     = "${var.project_name}-tg"
  port     = 80
  protocol = "HTTP"
  vpc_id   = module.vpc.vpc_id

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 10
  }
}

resource "aws_lb_listener" "web" {
  load_balancer_arn = aws_lb.web.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = aws_acm_certificate.main.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.web.arn
  }
}
```

### 6. RDS Database

```hcl
resource "aws_db_instance" "main" {
  identifier     = "${var.project_name}-db"
  engine         = "postgres"
  engine_version = "16.1"
  instance_class = var.db_instance_class

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true

  db_name  = var.db_name
  username = var.db_username
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "mon:04:00-mon:05:00"

  skip_final_snapshot = var.environment != "prod"
}
```

### 7. Variables

```hcl
# variables.tf
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod"
  }
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}
```

### 8. Outputs

```hcl
# outputs.tf
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "load_balancer_dns" {
  description = "Load balancer DNS name"
  value       = aws_lb.web.dns_name
}

output "database_endpoint" {
  description = "RDS endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}
```

## State Management

### Remote Backend (S3)

```hcl
# backend.tf
terraform {
  backend "s3" {
    bucket         = "mycompany-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"

    # State locking
    role_arn = "arn:aws:iam::123456789012:role/TerraformBackend"
  }
}
```

### State Commands

```bash
# List state resources
terraform state list

# Show resource details
terraform state show aws_instance.web

# Move resource
terraform state mv aws_instance.web aws_instance.web_server

# Remove resource from state
terraform state rm aws_instance.old

# Import existing resource
terraform import aws_instance.web i-1234567890abcdef0

# Pull remote state
terraform state pull > terraform.tfstate.backup

# Push local state
terraform state push terraform.tfstate
```

## Workspaces

```bash
# Create workspace
terraform workspace new staging

# List workspaces
terraform workspace list

# Select workspace
terraform workspace select prod

# Show current workspace
terraform workspace show

# Delete workspace
terraform workspace delete dev
```

## Drift Detection

```bash
# Detect drift (plan without changes)
terraform plan -detailed-exitcode

# Output:
# 0 = No changes
# 1 = Error
# 2 = Changes detected (drift)

# Automated drift detection
terraform plan -no-color | grep -q "No changes" && echo "No drift" || echo "Drift detected!"
```

## GitHub Actions Integration

```yaml
name: Terraform CI/CD

on:
  push:
    branches: [main]
    paths: ['terraform/**']
  pull_request:
    paths: ['terraform/**']

env:
  TF_VERSION: '1.5.0'

jobs:
  terraform:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: ./terraform

    steps:
    - uses: actions/checkout@v4

    - uses: hashicorp/setup-terraform@v3
      with:
        terraform_version: ${{ env.TF_VERSION }}

    - name: Configure AWS Credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1

    - name: Terraform Format
      run: terraform fmt -check

    - name: Terraform Init
      run: terraform init

    - name: Terraform Validate
      run: terraform validate

    - name: Terraform Plan
      run: terraform plan -out=tfplan

    - name: Terraform Apply
      if: github.ref == 'refs/heads/main' && github.event_name == 'push'
      run: terraform apply -auto-approve tfplan
```

## Best Practices

1. **Remote State**: Always use remote backend (S3, GCS, Azure Blob)
2. **State Locking**: Enable locking with DynamoDB or backend support
3. **Workspaces**: Separate environments (dev/staging/prod)
4. **Modules**: Reuse code with Terraform modules
5. **Version Pinning**: Pin Terraform and provider versions
6. **Secrets Management**: Use AWS Secrets Manager, Vault, or encrypted vars
7. **Plan Before Apply**: Always run `terraform plan` first
8. **Drift Detection**: Regularly check for infrastructure drift
9. **CI/CD Integration**: Automate with GitHub Actions/GitLab CI
10. **Tagging**: Tag all resources for cost tracking and management

## Security

```hcl
# Enable encryption
resource "aws_s3_bucket" "state" {
  bucket = "terraform-state"

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  versioning {
    enabled = true
  }
}

# Use IAM roles for authentication
provider "aws" {
  assume_role {
    role_arn = "arn:aws:iam::123456789012:role/TerraformRole"
  }
}
```

## Integration Points

- **/aws-deploy** - AWS deployment
- **/github-actions** - CI/CD automation
- **/ansible-deploy** - Configuration management
- **/docker-compose** - Container orchestration

## Agent Integration

```javascript
// Planner: Design infrastructure
mcp__ruv-swarm__agent_spawn({
  type: "researcher",
  capabilities: ["terraform", "aws-architecture"]
})

// Optimizer: Cost optimization
mcp__ruv-swarm__agent_spawn({
  type: "optimizer",
  capabilities: ["cloud-cost-optimization"]
})
```

## Related Commands

- `/aws-deploy` - AWS deployment
- `/ansible-deploy` - Configuration management
- `/github-actions` - CI/CD workflows

---

**Implementation Status**: Production-Ready
**Last Updated**: 2025-11-01
**Maintainer**: Integration Specialist
