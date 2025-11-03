# Multi-Cloud Infrastructure Configuration with Terraform
# Version: 2.0.0
# Supports: AWS, Azure, GCP

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
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Remote state backend with S3 + DynamoDB locking
  backend "s3" {
    bucket         = "terraform-state-${var.environment}"
    key            = "infrastructure/terraform.tfstate"
    region         = var.aws_region
    encrypt        = true
    dynamodb_table = "terraform-state-lock"

    # Optional: Use Azure Blob Storage
    # backend "azurerm" {
    #   resource_group_name  = "terraform-state-rg"
    #   storage_account_name = "tfstate${var.environment}"
    #   container_name       = "tfstate"
    #   key                  = "infrastructure.tfstate"
    # }
  }
}

# ==================== Variables ====================

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "azure_location" {
  description = "Azure location"
  type        = string
  default     = "East US"
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "gcp_project" {
  description = "GCP project ID"
  type        = string
}

variable "enable_aws" {
  description = "Enable AWS infrastructure"
  type        = bool
  default     = true
}

variable "enable_azure" {
  description = "Enable Azure infrastructure"
  type        = bool
  default     = false
}

variable "enable_gcp" {
  description = "Enable GCP infrastructure"
  type        = bool
  default     = false
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    ManagedBy = "Terraform"
  }
}

# ==================== Providers ====================

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = merge(var.tags, {
      Environment = var.environment
      Provider    = "AWS"
    })
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = true
    }
  }
}

provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

# ==================== AWS Infrastructure ====================

module "aws_infrastructure" {
  source = "./modules/aws"
  count  = var.enable_aws ? 1 : 0

  environment = var.environment
  vpc_cidr    = var.vpc_cidr
  region      = var.aws_region
}

# AWS VPC
resource "aws_vpc" "main" {
  count = var.enable_aws ? 1 : 0

  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.environment}-vpc"
  }
}

# AWS Public Subnet
resource "aws_subnet" "public" {
  count = var.enable_aws ? 2 : 0

  vpc_id                  = aws_vpc.main[0].id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available[0].names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.environment}-public-subnet-${count.index + 1}"
    Type = "public"
  }
}

# AWS Private Subnet
resource "aws_subnet" "private" {
  count = var.enable_aws ? 2 : 0

  vpc_id            = aws_vpc.main[0].id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available[0].names[count.index]

  tags = {
    Name = "${var.environment}-private-subnet-${count.index + 1}"
    Type = "private"
  }
}

# AWS Internet Gateway
resource "aws_internet_gateway" "main" {
  count = var.enable_aws ? 1 : 0

  vpc_id = aws_vpc.main[0].id

  tags = {
    Name = "${var.environment}-igw"
  }
}

# AWS NAT Gateway
resource "aws_eip" "nat" {
  count  = var.enable_aws ? 2 : 0
  domain = "vpc"

  tags = {
    Name = "${var.environment}-nat-eip-${count.index + 1}"
  }
}

resource "aws_nat_gateway" "main" {
  count = var.enable_aws ? 2 : 0

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = {
    Name = "${var.environment}-nat-${count.index + 1}"
  }
}

# AWS Security Group
resource "aws_security_group" "app" {
  count = var.enable_aws ? 1 : 0

  name        = "${var.environment}-app-sg"
  description = "Security group for application instances"
  vpc_id      = aws_vpc.main[0].id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.environment}-app-sg"
  }
}

# ==================== Azure Infrastructure ====================

resource "azurerm_resource_group" "main" {
  count = var.enable_azure ? 1 : 0

  name     = "${var.environment}-rg"
  location = var.azure_location

  tags = merge(var.tags, {
    Environment = var.environment
  })
}

resource "azurerm_virtual_network" "main" {
  count = var.enable_azure ? 1 : 0

  name                = "${var.environment}-vnet"
  address_space       = [var.vpc_cidr]
  location            = azurerm_resource_group.main[0].location
  resource_group_name = azurerm_resource_group.main[0].name
}

resource "azurerm_subnet" "public" {
  count = var.enable_azure ? 1 : 0

  name                 = "${var.environment}-public-subnet"
  resource_group_name  = azurerm_resource_group.main[0].name
  virtual_network_name = azurerm_virtual_network.main[0].name
  address_prefixes     = [cidrsubnet(var.vpc_cidr, 8, 0)]
}

resource "azurerm_subnet" "private" {
  count = var.enable_azure ? 1 : 0

  name                 = "${var.environment}-private-subnet"
  resource_group_name  = azurerm_resource_group.main[0].name
  virtual_network_name = azurerm_virtual_network.main[0].name
  address_prefixes     = [cidrsubnet(var.vpc_cidr, 8, 10)]
}

# ==================== GCP Infrastructure ====================

resource "google_compute_network" "main" {
  count = var.enable_gcp ? 1 : 0

  name                    = "${var.environment}-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "public" {
  count = var.enable_gcp ? 1 : 0

  name          = "${var.environment}-public-subnet"
  ip_cidr_range = cidrsubnet(var.vpc_cidr, 8, 0)
  region        = var.gcp_region
  network       = google_compute_network.main[0].id
}

resource "google_compute_subnetwork" "private" {
  count = var.enable_gcp ? 1 : 0

  name          = "${var.environment}-private-subnet"
  ip_cidr_range = cidrsubnet(var.vpc_cidr, 8, 10)
  region        = var.gcp_region
  network       = google_compute_network.main[0].id
}

# ==================== Data Sources ====================

data "aws_availability_zones" "available" {
  count = var.enable_aws ? 1 : 0

  state = "available"
}

# ==================== Outputs ====================

output "aws_vpc_id" {
  description = "AWS VPC ID"
  value       = var.enable_aws ? aws_vpc.main[0].id : null
}

output "aws_public_subnet_ids" {
  description = "AWS public subnet IDs"
  value       = var.enable_aws ? aws_subnet.public[*].id : []
}

output "aws_private_subnet_ids" {
  description = "AWS private subnet IDs"
  value       = var.enable_aws ? aws_subnet.private[*].id : []
}

output "azure_resource_group_name" {
  description = "Azure resource group name"
  value       = var.enable_azure ? azurerm_resource_group.main[0].name : null
}

output "azure_vnet_id" {
  description = "Azure VNet ID"
  value       = var.enable_azure ? azurerm_virtual_network.main[0].id : null
}

output "gcp_network_name" {
  description = "GCP network name"
  value       = var.enable_gcp ? google_compute_network.main[0].name : null
}

output "gcp_subnet_ids" {
  description = "GCP subnet IDs"
  value = var.enable_gcp ? {
    public  = google_compute_subnetwork.public[0].id
    private = google_compute_subnetwork.private[0].id
  } : null
}
