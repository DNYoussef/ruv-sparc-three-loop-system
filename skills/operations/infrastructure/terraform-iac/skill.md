---
name: terraform-iac
description: Terraform infrastructure as code specialist for multi-cloud deployments (AWS/GCP/Azure), state management with remote backends, module development, drift detection, policy as code with Sentinel/OPA, and GitOps workflows. Use when provisioning cloud infrastructure, managing infrastructure state, implementing IaC best practices, or requiring Terraform expertise. Handles workspaces, dynamic blocks, for_each loops, and production-grade Terraform configurations.
category: Infrastructure
complexity: High
triggers: ["terraform", "iac", "infrastructure as code", "terraform state", "terraform modules", "multi-cloud", "terraform plan", "terraform apply", "hcl"]
---

# Terraform Infrastructure as Code Specialist

Expert Terraform for cloud-agnostic infrastructure provisioning and state management.

## Purpose

Comprehensive Terraform expertise including multi-cloud deployments, state management, module development, drift detection, and GitOps. Ensures infrastructure is versioned, reproducible, and maintainable.

## When to Use

- Provisioning cloud infrastructure (AWS, GCP, Azure)
- Managing infrastructure state with remote backends
- Creating reusable Terraform modules
- Implementing GitOps for infrastructure
- Detecting and fixing infrastructure drift
- Migrating from manual infrastructure to IaC
- Multi-environment deployments (dev, staging, prod)

## Prerequisites

**Required**: Cloud provider basics (AWS/GCP/Azure), HCL syntax, Terraform CLI

**Agents**: `system-architect`, `cicd-engineer`, `security-manager`, `reviewer`

## Core Workflows

### Workflow 1: AWS Infrastructure with Modules

**Step 1: Directory Structure**

```
terraform/
├── main.tf
├── variables.tf
├── outputs.tf
├── versions.tf
├── terraform.tfvars
└── modules/
    ├── vpc/
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    └── ec2/
        ├── main.tf
        ├── variables.tf
        └── outputs.tf
```

**Step 2: Main Configuration**

```hcl
# main.tf
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-lock"
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

module "vpc" {
  source = "./modules/vpc"

  vpc_cidr           = var.vpc_cidr
  availability_zones = var.availability_zones
  environment        = var.environment
}

module "ec2" {
  source = "./modules/ec2"

  vpc_id            = module.vpc.vpc_id
  subnet_ids        = module.vpc.private_subnet_ids
  instance_type     = var.instance_type
  instance_count    = var.instance_count
  security_group_id = module.vpc.security_group_id
}
```

**Step 3: Variables and Outputs**

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
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# outputs.tf
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "instance_ids" {
  description = "EC2 instance IDs"
  value       = module.ec2.instance_ids
}
```

### Workflow 2: Dynamic Blocks and for_each

```hcl
# Dynamic ingress rules
resource "aws_security_group" "app" {
  name   = "${var.environment}-app-sg"
  vpc_id = var.vpc_id

  dynamic "ingress" {
    for_each = var.ingress_rules
    content {
      from_port   = ingress.value.from_port
      to_port     = ingress.value.to_port
      protocol    = ingress.value.protocol
      cidr_blocks = ingress.value.cidr_blocks
      description = ingress.value.description
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# for_each for multiple resources
resource "aws_instance" "app" {
  for_each = toset(var.availability_zones)

  ami           = var.ami_id
  instance_type = var.instance_type
  subnet_id     = var.subnet_ids[each.key]

  tags = {
    Name = "${var.environment}-app-${each.key}"
    AZ   = each.value
  }
}
```

### Workflow 3: Remote State and Data Sources

```hcl
# Reference remote state from another workspace
data "terraform_remote_state" "vpc" {
  backend = "s3"
  config = {
    bucket = "my-terraform-state"
    key    = "network/terraform.tfstate"
    region = "us-east-1"
  }
}

# Use outputs from remote state
resource "aws_instance" "app" {
  subnet_id = data.terraform_remote_state.vpc.outputs.private_subnet_ids[0]
  # ...
}

# Data source for existing resources
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}
```

### Workflow 4: Workspaces for Environments

```bash
# Create workspaces
terraform workspace new dev
terraform workspace new staging
terraform workspace new production

# List workspaces
terraform workspace list

# Switch workspace
terraform workspace select production

# Use workspace in configuration
resource "aws_instance" "app" {
  instance_type = terraform.workspace == "production" ? "t3.large" : "t3.micro"

  tags = {
    Environment = terraform.workspace
  }
}
```

### Workflow 5: GitOps Workflow

```bash
# 1. Initialize Terraform
terraform init

# 2. Format code
terraform fmt -recursive

# 3. Validate configuration
terraform validate

# 4. Plan changes
terraform plan -out=tfplan

# 5. Review plan
terraform show tfplan

# 6. Apply changes (in CI/CD)
terraform apply tfplan

# 7. Check for drift
terraform plan -detailed-exitcode
# Exit code 2 means drift detected
```

**GitHub Actions CI/CD**

```yaml
# .github/workflows/terraform.yml
name: Terraform
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  terraform:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2
        with:
          terraform_version: 1.5.0

      - name: Terraform Init
        run: terraform init

      - name: Terraform Format
        run: terraform fmt -check -recursive

      - name: Terraform Validate
        run: terraform validate

      - name: Terraform Plan
        run: terraform plan -no-color
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Terraform Apply
        if: github.ref == 'refs/heads/main' && github.event_name == 'push'
        run: terraform apply -auto-approve
```

## Best Practices

**1. Remote State with Locking**
```hcl
# ✅ GOOD: S3 backend with DynamoDB locking
terraform {
  backend "s3" {
    bucket         = "terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-lock"  # Prevents concurrent modifications
  }
}

# ❌ BAD: Local state (not suitable for teams)
terraform {
  backend "local" {
    path = "terraform.tfstate"
  }
}
```

**2. Use Modules for Reusability**
```hcl
# ✅ GOOD: Reusable module
module "web_app" {
  source = "terraform-aws-modules/ec2-instance/aws"
  version = "5.0.0"

  name = "web-app"
  # ...
}

# ❌ BAD: Copy-pasting resource definitions
```

**3. Variables with Validation**
```hcl
variable "instance_type" {
  type        = string
  default     = "t3.micro"

  validation {
    condition     = can(regex("^t3\\.", var.instance_type))
    error_message = "Instance type must be from the t3 family."
  }
}
```

**4. Sensitive Data**
```hcl
# Mark sensitive outputs
output "db_password" {
  value     = aws_db_instance.main.password
  sensitive = true
}

# Use AWS Secrets Manager
data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = "prod/db/password"
}

resource "aws_db_instance" "main" {
  password = data.aws_secretsmanager_secret_version.db_password.secret_string
  # ...
}
```

**5. Tagging Strategy**
```hcl
locals {
  common_tags = {
    Environment = var.environment
    ManagedBy   = "Terraform"
    Project     = var.project_name
    CostCenter  = var.cost_center
  }
}

resource "aws_instance" "app" {
  tags = merge(local.common_tags, {
    Name = "app-server"
  })
}
```

## Quality Criteria

- ✅ Remote state with locking configured
- ✅ All resources tagged consistently
- ✅ Modules used for reusable components
- ✅ Variables have descriptions and types
- ✅ Sensitive data marked as sensitive
- ✅ terraform fmt passes
- ✅ terraform validate passes
- ✅ No hardcoded credentials

## Common Commands

```bash
# Initialize
terraform init

# Plan changes
terraform plan
terraform plan -out=tfplan

# Apply changes
terraform apply
terraform apply tfplan
terraform apply -auto-approve

# Destroy infrastructure
terraform destroy

# Format code
terraform fmt -recursive

# Validate configuration
terraform validate

# Show current state
terraform show

# List resources
terraform state list

# Import existing resource
terraform import aws_instance.app i-1234567890abcdef0

# Refresh state
terraform refresh

# Check for drift
terraform plan -detailed-exitcode
```

## Troubleshooting

**Issue**: State lock timeout
**Solution**: Check DynamoDB table for stuck locks, force-unlock with caution: `terraform force-unlock <LOCK_ID>`

**Issue**: Resource already exists
**Solution**: Import existing resource: `terraform import <resource_type>.<name> <id>`

**Issue**: Drift detected
**Solution**: Review `terraform plan`, update code or manually fix infrastructure

## Related Skills

- `aws-specialist`: AWS-specific resources
- `kubernetes-specialist`: EKS clusters with Terraform
- `docker-containerization`: Infrastructure for containers

## Tools

- Terraform Cloud: SaaS for state management
- Terragrunt: Terraform wrapper for DRY configs
- tfsec: Security scanner for Terraform
- Checkov: Policy-as-code scanner
- Infracost: Cost estimation for Terraform

## MCP Tools

- `mcp__flow-nexus__sandbox_execute` for Terraform commands
- `mcp__memory-mcp__memory_store` for IaC patterns

## Success Metrics

- Infrastructure drift: 0%
- Deployment time: <15 minutes
- Code reuse: ≥70% via modules
- Test coverage: 100% of modules

---

**Skill Version**: 1.0.0
**Last Updated**: 2025-11-02
