# Terraform Infrastructure as Code Best Practices

Comprehensive guide to building, managing, and scaling cloud infrastructure with Terraform for multi-cloud deployments.

## Why Terraform?

**Key Benefits**:
- **Multi-cloud support**: AWS, GCP, Azure, and 3,000+ providers
- **Declarative syntax**: Describe desired state, not imperative steps
- **State management**: Track infrastructure state and detect drift
- **Plan before apply**: Preview changes before execution
- **Version control**: Infrastructure as code in Git
- **Reusable modules**: DRY principle for infrastructure
- **Community ecosystem**: Terraform Registry with 3,000+ modules

## Project Structure

### Recommended Layout

```
terraform/
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── terraform.tfvars
│   │   └── backend.tf
│   ├── staging/
│   └── production/
├── modules/
│   ├── vpc/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── README.md
│   ├── compute/
│   └── database/
├── global/
│   ├── iam/
│   └── route53/
└── README.md
```

**Key Principles**:
1. **Separate environments** into different directories
2. **Create reusable modules** for common patterns
3. **Use remote backends** for state management
4. **Version control everything** except secrets
5. **Document modules** with README.md

---

## Core Concepts

### Providers

Providers are plugins that interact with cloud platforms and services.

```hcl
terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      ManagedBy   = "terraform"
      Project     = var.project_name
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

provider "azurerm" {
  features {}
}
```

### Resources

Resources are the most important element - they define infrastructure components.

```hcl
resource "aws_instance" "web" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.micro"

  vpc_security_group_ids = [aws_security_group.web.id]
  subnet_id              = aws_subnet.public[0].id

  user_data = file("${path.module}/scripts/install.sh")

  tags = {
    Name = "web-server-${var.environment}"
  }
}
```

### Data Sources

Data sources query existing infrastructure or external APIs.

```hcl
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]  # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}
```

### Variables

Variables parameterize configurations for reusability.

**variables.tf**:
```hcl
variable "environment" {
  description = "Environment name (dev, staging, production)"
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

variable "instance_count" {
  description = "Number of EC2 instances"
  type        = number
  default     = 2

  validation {
    condition     = var.instance_count >= 1 && var.instance_count <= 10
    error_message = "Instance count must be between 1 and 10."
  }
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}
```

**terraform.tfvars**:
```hcl
environment    = "production"
vpc_cidr       = "10.0.0.0/16"
instance_count = 3

tags = {
  Owner       = "platform-team"
  CostCenter  = "engineering"
}
```

### Outputs

Outputs expose values for use by other Terraform configurations or external tools.

**outputs.tf**:
```hcl
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = aws_subnet.public[*].id
}

output "web_server_public_ip" {
  description = "Public IP of web server"
  value       = aws_instance.web.public_ip
  sensitive   = true  # Hide from console output
}

output "database_endpoint" {
  description = "RDS database endpoint"
  value       = aws_db_instance.main.endpoint
}
```

---

## State Management

### Remote Backend

**NEVER store state files in Git**. Use remote backends for:
- **Team collaboration**: Shared state
- **State locking**: Prevent concurrent modifications
- **Encryption**: Secure sensitive data
- **Versioning**: State history

**S3 Backend (AWS)**:
```hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}
```

**GCS Backend (GCP)**:
```hcl
terraform {
  backend "gcs" {
    bucket = "my-terraform-state"
    prefix = "production"
  }
}
```

**Azure Storage Backend**:
```hcl
terraform {
  backend "azurerm" {
    resource_group_name  = "terraform-state-rg"
    storage_account_name = "tfstate"
    container_name       = "tfstate"
    key                  = "production.tfstate"
  }
}
```

**Terraform Cloud Backend**:
```hcl
terraform {
  cloud {
    organization = "my-org"

    workspaces {
      name = "production"
    }
  }
}
```

### State Locking

DynamoDB table for S3 backend locking:

```hcl
resource "aws_dynamodb_table" "terraform_locks" {
  name         = "terraform-locks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  tags = {
    Name = "Terraform State Lock Table"
  }
}
```

---

## Modules

Modules are containers for multiple resources used together.

### Creating a Module

**modules/vpc/main.tf**:
```hcl
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(
    var.tags,
    {
      Name = "${var.name}-vpc"
    }
  )
}

resource "aws_subnet" "public" {
  count = length(var.public_subnet_cidrs)

  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = merge(
    var.tags,
    {
      Name = "${var.name}-public-${count.index + 1}"
      Type = "public"
    }
  )
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = merge(
    var.tags,
    {
      Name = "${var.name}-igw"
    }
  )
}
```

**modules/vpc/variables.tf**:
```hcl
variable "name" {
  description = "Name prefix for VPC resources"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
```

**modules/vpc/outputs.tf**:
```hcl
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = aws_subnet.public[*].id
}

output "internet_gateway_id" {
  description = "ID of the internet gateway"
  value       = aws_internet_gateway.main.id
}
```

### Using a Module

**environments/production/main.tf**:
```hcl
module "vpc" {
  source = "../../modules/vpc"

  name                = "production"
  vpc_cidr            = "10.0.0.0/16"
  public_subnet_cidrs = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]

  tags = {
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

module "compute" {
  source = "../../modules/compute"

  vpc_id            = module.vpc.vpc_id
  subnet_ids        = module.vpc.public_subnet_ids
  instance_count    = 3
  instance_type     = "t3.medium"
}
```

### Terraform Registry Modules

Use community modules from the Terraform Registry:

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.1.0"

  name = "production-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  enable_vpn_gateway = false

  tags = {
    Environment = "production"
  }
}
```

---

## Best Practices

### 1. Use Workspaces for Environments

```bash
# Create workspace
terraform workspace new production

# List workspaces
terraform workspace list

# Switch workspace
terraform workspace select production

# Use workspace in code
resource "aws_instance" "web" {
  instance_type = terraform.workspace == "production" ? "t3.large" : "t3.micro"

  tags = {
    Environment = terraform.workspace
  }
}
```

### 2. Implement Resource Naming Conventions

```hcl
locals {
  name_prefix = "${var.project_name}-${var.environment}"

  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Owner       = var.owner
  }
}

resource "aws_instance" "web" {
  # ... other config

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-web-${count.index + 1}"
    }
  )
}
```

### 3. Use Data Sources for Dynamic Values

```hcl
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

resource "aws_subnet" "public" {
  count = length(data.aws_availability_zones.available.names)

  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Region    = data.aws_region.current.name
    AccountId = data.aws_caller_identity.current.account_id
  }
}
```

### 4. Implement Lifecycle Rules

```hcl
resource "aws_instance" "web" {
  # ... other config

  lifecycle {
    create_before_destroy = true
    prevent_destroy       = true
    ignore_changes        = [
      user_data,
      tags["UpdatedAt"]
    ]
  }
}
```

### 5. Use Locals for Complex Expressions

```hcl
locals {
  subnet_count = length(data.aws_availability_zones.available.names)

  vpc_cidr_blocks = [
    for i in range(local.subnet_count) :
    cidrsubnet(var.vpc_cidr, 8, i)
  ]

  environment_config = {
    dev = {
      instance_type = "t3.micro"
      instance_count = 1
    }
    production = {
      instance_type = "t3.large"
      instance_count = 3
    }
  }

  instance_type = local.environment_config[var.environment].instance_type
}
```

### 6. Implement Conditional Resources

```hcl
resource "aws_instance" "bastion" {
  count = var.create_bastion ? 1 : 0

  # ... config
}

resource "aws_db_instance" "replica" {
  count = var.environment == "production" ? 1 : 0

  replicate_source_db = aws_db_instance.main.id
  # ... config
}
```

### 7. Use Dynamic Blocks

```hcl
resource "aws_security_group" "web" {
  name_prefix = "${var.name}-web-"
  vpc_id      = var.vpc_id

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
}
```

---

## Multi-Cloud Architecture

### Unified Multi-Cloud Module

```hcl
module "compute" {
  source = "./modules/compute"

  provider_type = var.provider_type  # "aws", "gcp", or "azure"

  # Common parameters
  name            = var.name
  instance_count  = var.instance_count
  instance_type   = var.instance_type

  # Provider-specific parameters
  aws_ami         = var.aws_ami
  gcp_image       = var.gcp_image
  azure_image     = var.azure_image
}
```

### Provider Aliases for Multi-Region

```hcl
provider "aws" {
  alias  = "us_east"
  region = "us-east-1"
}

provider "aws" {
  alias  = "eu_west"
  region = "eu-west-1"
}

module "vpc_us" {
  source = "./modules/vpc"

  providers = {
    aws = aws.us_east
  }
}

module "vpc_eu" {
  source = "./modules/vpc"

  providers = {
    aws = aws.eu_west
  }
}
```

---

## CI/CD Integration

### GitHub Actions

**.github/workflows/terraform.yml**:
```yaml
name: Terraform

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  terraform:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2
        with:
          terraform_version: 1.6.0

      - name: Terraform Format
        run: terraform fmt -check -recursive

      - name: Terraform Init
        run: terraform init
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Terraform Validate
        run: terraform validate

      - name: Terraform Plan
        run: terraform plan -out=tfplan
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Terraform Apply
        if: github.ref == 'refs/heads/main' && github.event_name == 'push'
        run: terraform apply -auto-approve tfplan
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

---

## Troubleshooting

### Common Issues

1. **State Lock Errors**:
```bash
# Force unlock (use with caution)
terraform force-unlock <LOCK_ID>
```

2. **Drift Detection**:
```bash
# Detect configuration drift
terraform plan -detailed-exitcode

# Refresh state without applying
terraform refresh
```

3. **Import Existing Resources**:
```bash
# Import existing AWS EC2 instance
terraform import aws_instance.web i-1234567890abcdef0
```

4. **Debug Mode**:
```bash
# Enable debug logging
export TF_LOG=DEBUG
terraform apply
```

---

## Related Resources

- [Terraform Documentation](https://www.terraform.io/docs)
- [Terraform Registry](https://registry.terraform.io/)
- [Terraform Best Practices](https://www.terraform-best-practices.com/)
- [HashiCorp Learn](https://learn.hashicorp.com/terraform)
