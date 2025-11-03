# Terraform Multi-Cloud Infrastructure Example

**Scenario**: Deploy multi-cloud infrastructure across AWS and Azure using Terraform with remote state management, modular design, GitOps workflow, security scanning (Checkov, tfsec), and automated CI/CD integration.

**Components**: AWS VPC + EC2, Azure VNet + VMs, S3 remote state, DynamoDB state locking, Terraform modules, CI/CD pipeline, security compliance

**Lines**: ~270

---

## Prerequisites

```bash
# Required tools
terraform --version  # Terraform 1.5+
aws --version        # AWS CLI 2.0+
az --version         # Azure CLI 2.40+
tfsec --version      # tfsec for security scanning
checkov --version    # Checkov for compliance checking

# Install tfsec and checkov
brew install tfsec checkov  # macOS
# OR
curl -s https://raw.githubusercontent.com/aquasecurity/tfsec/master/scripts/install_linux.sh | bash
pip install checkov
```

## Project Structure

```
terraform-infra/
├── main.tf                    # Root module
├── variables.tf               # Variable definitions
├── outputs.tf                 # Output values
├── terraform.tf               # Provider and backend config
├── versions.tf                # Provider version constraints
├── environments/
│   ├── dev/
│   │   ├── terraform.tfvars
│   │   └── backend.tf
│   ├── staging/
│   │   ├── terraform.tfvars
│   │   └── backend.tf
│   └── production/
│       ├── terraform.tfvars
│       └── backend.tf
├── modules/
│   ├── aws-vpc/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── aws-compute/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── azure-network/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── azure-compute/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
├── scripts/
│   ├── terraform-plan.sh
│   ├── terraform-apply.sh
│   └── security-scan.sh
└── .github/
    └── workflows/
        └── terraform-ci.yml
```

## Step 1: Configure Remote State Backend

Create S3 bucket and DynamoDB table for state:

```bash
# Create S3 bucket for Terraform state
aws s3api create-bucket \
  --bucket terraform-state-production-12345 \
  --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket terraform-state-production-12345 \
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
  --bucket terraform-state-production-12345 \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'

# Block public access
aws s3api put-public-access-block \
  --bucket terraform-state-production-12345 \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# Create DynamoDB table for state locking
aws dynamodb create-table \
  --table-name terraform-state-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
  --region us-east-1
```

Create `terraform.tf`:

```hcl
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }

  backend "s3" {
    bucket         = "terraform-state-production-12345"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"

    # Workspace-specific state files
    workspace_key_prefix = "workspaces"
  }
}
```

## Step 2: Create AWS VPC Module

Create `modules/aws-vpc/main.tf`:

```hcl
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(
    var.tags,
    {
      Name = "${var.environment}-vpc"
    }
  )
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = merge(
    var.tags,
    {
      Name = "${var.environment}-igw"
    }
  )
}

# Public Subnets
resource "aws_subnet" "public" {
  count = length(var.availability_zones)

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = merge(
    var.tags,
    {
      Name = "${var.environment}-public-subnet-${count.index + 1}"
      Type = "public"
      "kubernetes.io/role/elb" = "1"  # For EKS load balancers
    }
  )
}

# Private Subnets
resource "aws_subnet" "private" {
  count = length(var.availability_zones)

  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = var.availability_zones[count.index]

  tags = merge(
    var.tags,
    {
      Name = "${var.environment}-private-subnet-${count.index + 1}"
      Type = "private"
      "kubernetes.io/role/internal-elb" = "1"
    }
  )
}

# NAT Gateways
resource "aws_eip" "nat" {
  count  = var.enable_nat_gateway ? length(var.availability_zones) : 0
  domain = "vpc"

  tags = merge(
    var.tags,
    {
      Name = "${var.environment}-nat-eip-${count.index + 1}"
    }
  )

  depends_on = [aws_internet_gateway.main]
}

resource "aws_nat_gateway" "main" {
  count = var.enable_nat_gateway ? length(var.availability_zones) : 0

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = merge(
    var.tags,
    {
      Name = "${var.environment}-nat-${count.index + 1}"
    }
  )

  depends_on = [aws_internet_gateway.main]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.environment}-public-rt"
    }
  )
}

resource "aws_route_table" "private" {
  count = var.enable_nat_gateway ? length(var.availability_zones) : 0

  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.environment}-private-rt-${count.index + 1}"
    }
  )
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = var.enable_nat_gateway ? length(aws_subnet.private) : 0

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# VPC Flow Logs
resource "aws_flow_log" "main" {
  count = var.enable_flow_logs ? 1 : 0

  iam_role_arn    = aws_iam_role.flow_logs[0].arn
  log_destination = aws_cloudwatch_log_group.flow_logs[0].arn
  traffic_type    = "ALL"
  vpc_id          = aws_vpc.main.id
}

resource "aws_cloudwatch_log_group" "flow_logs" {
  count = var.enable_flow_logs ? 1 : 0

  name              = "/aws/vpc/${var.environment}"
  retention_in_days = 30

  tags = var.tags
}

resource "aws_iam_role" "flow_logs" {
  count = var.enable_flow_logs ? 1 : 0

  name = "${var.environment}-vpc-flow-logs"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "vpc-flow-logs.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "flow_logs" {
  count = var.enable_flow_logs ? 1 : 0

  name = "${var.environment}-vpc-flow-logs"
  role = aws_iam_role.flow_logs[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogGroups",
        "logs:DescribeLogStreams"
      ]
      Effect   = "Allow"
      Resource = "*"
    }]
  })
}
```

Create `modules/aws-vpc/variables.tf`:

```hcl
variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
}

variable "enable_nat_gateway" {
  description = "Enable NAT gateway for private subnets"
  type        = bool
  default     = true
}

variable "enable_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default     = {}
}
```

Create `modules/aws-vpc/outputs.tf`:

```hcl
output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "nat_gateway_ips" {
  description = "NAT gateway elastic IPs"
  value       = aws_eip.nat[*].public_ip
}
```

## Step 3: Create Azure Network Module

Create `modules/azure-network/main.tf`:

```hcl
resource "azurerm_resource_group" "main" {
  name     = "${var.environment}-rg"
  location = var.location

  tags = var.tags
}

resource "azurerm_virtual_network" "main" {
  name                = "${var.environment}-vnet"
  address_space       = [var.vnet_cidr]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  tags = var.tags
}

resource "azurerm_subnet" "public" {
  name                 = "${var.environment}-public-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_cidr, 8, 0)]
}

resource "azurerm_subnet" "private" {
  name                 = "${var.environment}-private-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_cidr, 8, 10)]

  service_endpoints = [
    "Microsoft.Storage",
    "Microsoft.Sql",
    "Microsoft.KeyVault"
  ]
}

resource "azurerm_network_security_group" "public" {
  name                = "${var.environment}-public-nsg"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  security_rule {
    name                       = "AllowHTTP"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "80"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "AllowHTTPS"
    priority                   = 110
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  tags = var.tags
}

resource "azurerm_subnet_network_security_group_association" "public" {
  subnet_id                 = azurerm_subnet.public.id
  network_security_group_id = azurerm_network_security_group.public.id
}
```

## Step 4: Root Module Configuration

Create `main.tf`:

```hcl
# AWS Infrastructure
module "aws_vpc" {
  source = "./modules/aws-vpc"

  environment        = var.environment
  vpc_cidr           = var.aws_vpc_cidr
  availability_zones = var.aws_availability_zones
  enable_nat_gateway = var.enable_nat_gateway
  enable_flow_logs   = var.enable_flow_logs

  tags = local.common_tags
}

# Azure Infrastructure
module "azure_network" {
  source = "./modules/azure-network"

  environment = var.environment
  location    = var.azure_location
  vnet_cidr   = var.azure_vnet_cidr

  tags = local.common_tags
}

# Local values
locals {
  common_tags = merge(
    var.tags,
    {
      Environment = var.environment
      ManagedBy   = "Terraform"
      Project     = var.project_name
      CreatedAt   = timestamp()
    }
  )
}
```

## Step 5: Variables and Outputs

Create `variables.tf`:

```hcl
variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "multi-cloud-infra"
}

variable "aws_vpc_cidr" {
  description = "AWS VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "aws_availability_zones" {
  description = "AWS availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "azure_location" {
  description = "Azure location"
  type        = string
  default     = "East US"
}

variable "azure_vnet_cidr" {
  description = "Azure VNet CIDR block"
  type        = string
  default     = "10.1.0.0/16"
}

variable "enable_nat_gateway" {
  description = "Enable NAT gateway"
  type        = bool
  default     = true
}

variable "enable_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Common tags"
  type        = map(string)
  default     = {}
}
```

Create `outputs.tf`:

```hcl
output "aws_vpc_id" {
  description = "AWS VPC ID"
  value       = module.aws_vpc.vpc_id
}

output "aws_public_subnets" {
  description = "AWS public subnet IDs"
  value       = module.aws_vpc.public_subnet_ids
}

output "aws_private_subnets" {
  description = "AWS private subnet IDs"
  value       = module.aws_vpc.private_subnet_ids
}

output "azure_resource_group_name" {
  description = "Azure resource group name"
  value       = module.azure_network.resource_group_name
}

output "azure_vnet_id" {
  description = "Azure VNet ID"
  value       = module.azure_network.vnet_id
}
```

## Step 6: Environment-Specific Configuration

Create `environments/production/terraform.tfvars`:

```hcl
environment = "production"
project_name = "multi-cloud-infra"

# AWS Configuration
aws_vpc_cidr = "10.0.0.0/16"
aws_availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
enable_nat_gateway = true
enable_flow_logs = true

# Azure Configuration
azure_location = "East US"
azure_vnet_cidr = "10.1.0.0/16"

# Tags
tags = {
  CostCenter = "Engineering"
  Owner      = "DevOps Team"
  Compliance = "SOC2"
}
```

## Step 7: Security Scanning Script

Create `scripts/security-scan.sh`:

```bash
#!/bin/bash
set -euo pipefail

echo "Running Terraform security scans..."

# Run tfsec
echo "Running tfsec..."
tfsec . --format lovely --minimum-severity MEDIUM

# Run Checkov
echo "Running Checkov..."
checkov -d . --framework terraform --output cli --soft-fail

# Run Terraform validate
echo "Running Terraform validate..."
terraform init -backend=false
terraform validate

echo "Security scan completed successfully!"
```

## Step 8: CI/CD Pipeline

Create `.github/workflows/terraform-ci.yml`:

```yaml
name: Terraform CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  TF_VERSION: 1.5.0
  AWS_REGION: us-east-1

jobs:
  validate:
    name: Validate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Terraform Format Check
        run: terraform fmt -check -recursive

      - name: Terraform Init
        run: terraform init -backend=false

      - name: Terraform Validate
        run: terraform validate

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run tfsec
        uses: aquasecurity/tfsec-action@v1.0.0

      - name: Run Checkov
        uses: bridgecrewio/checkov-action@master
        with:
          directory: .
          framework: terraform

  plan:
    name: Plan
    runs-on: ubuntu-latest
    needs: [validate, security]
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Terraform Init
        run: terraform init

      - name: Terraform Plan
        run: terraform plan -var-file=environments/production/terraform.tfvars -out=tfplan

      - name: Upload Plan
        uses: actions/upload-artifact@v3
        with:
          name: tfplan
          path: tfplan

  apply:
    name: Apply
    runs-on: ubuntu-latest
    needs: plan
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Terraform Init
        run: terraform init

      - name: Download Plan
        uses: actions/download-artifact@v3
        with:
          name: tfplan

      - name: Terraform Apply
        run: terraform apply -auto-approve tfplan
```

## Step 9: Deployment Workflow

```bash
# Initialize Terraform
terraform init

# Select workspace
terraform workspace new production
terraform workspace select production

# Format code
terraform fmt -recursive

# Validate configuration
terraform validate

# Run security scans
./scripts/security-scan.sh

# Plan changes
terraform plan -var-file=environments/production/terraform.tfvars -out=tfplan

# Review plan output carefully

# Apply changes
terraform apply tfplan

# View outputs
terraform output

# Generate dependency graph
terraform graph | dot -Tsvg > graph.svg
```

## Step 10: State Management

```bash
# List workspaces
terraform workspace list

# View state
terraform show

# List resources
terraform state list

# Move resources between states
terraform state mv aws_vpc.main module.aws_vpc.aws_vpc.main

# Import existing resources
terraform import module.aws_vpc.aws_vpc.main vpc-12345678

# Remove resources from state (without destroying)
terraform state rm module.azure_network.azurerm_resource_group.main
```

## Best Practices

1. **Always use remote state** with locking
2. **Use modules** for reusability
3. **Run security scans** before applying
4. **Use workspaces** for environment isolation
5. **Tag all resources** for cost allocation
6. **Use variables** and tfvars files
7. **Document outputs** clearly
8. **Version pin** providers
9. **Enable deletion protection** on critical resources
10. **Regular state backups**

---

**Result**: Production-ready multi-cloud infrastructure with:
- Modular, reusable code
- Remote state with locking
- Security scanning (tfsec, Checkov)
- CI/CD automation
- Multi-environment support
- GitOps workflow
- Compliance validation
- Cost tagging
