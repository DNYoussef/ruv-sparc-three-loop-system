#!/bin/bash
# Infrastructure Provisioning Automation Script
# Purpose: Automate cloud resource provisioning across AWS, Azure, and GCP
# Version: 2.0.0
# Last Updated: 2025-11-02

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-/var/log/infra-provisioner.log}"
STATE_DIR="${STATE_DIR:-${HOME}/.infra-provisioner}"
TERRAFORM_VERSION="${TERRAFORM_VERSION:-1.5.0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

# Initialize state directory
init_state_dir() {
    log_info "Initializing state directory: $STATE_DIR"
    mkdir -p "$STATE_DIR"/{aws,azure,gcp,terraform}
    log_success "State directory initialized"
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."

    local missing_tools=()

    # Check for required tools
    for tool in terraform aws azure gcloud docker kubectl; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        return 1
    fi

    # Validate Terraform version
    local tf_version
    tf_version=$(terraform version -json | jq -r '.terraform_version')
    log_info "Terraform version: $tf_version"

    # Check cloud provider credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_warning "AWS credentials not configured or invalid"
    else
        log_success "AWS credentials validated"
    fi

    if ! az account show &> /dev/null; then
        log_warning "Azure credentials not configured or invalid"
    else
        log_success "Azure credentials validated"
    fi

    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        log_warning "GCP credentials not configured or invalid"
    else
        log_success "GCP credentials validated"
    fi

    log_success "Prerequisites validated"
}

# AWS provisioning functions
provision_aws() {
    local environment=$1
    local region=${2:-us-east-1}

    log_info "Provisioning AWS infrastructure in $region for $environment"

    # Export AWS environment variables
    export AWS_DEFAULT_REGION="$region"

    # Create VPC
    local vpc_id
    vpc_id=$(aws ec2 create-vpc \
        --cidr-block "10.0.0.0/16" \
        --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=${environment}-vpc},{Key=Environment,Value=${environment}}]" \
        --query 'Vpc.VpcId' \
        --output text)

    log_info "Created VPC: $vpc_id"

    # Create subnets
    local subnet_public subnet_private
    subnet_public=$(aws ec2 create-subnet \
        --vpc-id "$vpc_id" \
        --cidr-block "10.0.1.0/24" \
        --availability-zone "${region}a" \
        --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=${environment}-public-subnet},{Key=Type,Value=public}]" \
        --query 'Subnet.SubnetId' \
        --output text)

    subnet_private=$(aws ec2 create-subnet \
        --vpc-id "$vpc_id" \
        --cidr-block "10.0.2.0/24" \
        --availability-zone "${region}b" \
        --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=${environment}-private-subnet},{Key=Type,Value=private}]" \
        --query 'Subnet.SubnetId' \
        --output text)

    log_info "Created public subnet: $subnet_public"
    log_info "Created private subnet: $subnet_private"

    # Save state
    cat > "$STATE_DIR/aws/${environment}.json" <<EOF
{
  "vpc_id": "$vpc_id",
  "subnet_public": "$subnet_public",
  "subnet_private": "$subnet_private",
  "region": "$region",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

    log_success "AWS infrastructure provisioned for $environment"
}

# Azure provisioning functions
provision_azure() {
    local environment=$1
    local location=${2:-eastus}
    local resource_group="${environment}-rg"

    log_info "Provisioning Azure infrastructure in $location for $environment"

    # Create resource group
    az group create \
        --name "$resource_group" \
        --location "$location" \
        --tags Environment="$environment" \
        > /dev/null

    log_info "Created resource group: $resource_group"

    # Create virtual network
    az network vnet create \
        --resource-group "$resource_group" \
        --name "${environment}-vnet" \
        --address-prefix "10.1.0.0/16" \
        --subnet-name "${environment}-subnet" \
        --subnet-prefix "10.1.1.0/24" \
        > /dev/null

    log_info "Created virtual network: ${environment}-vnet"

    # Save state
    cat > "$STATE_DIR/azure/${environment}.json" <<EOF
{
  "resource_group": "$resource_group",
  "vnet": "${environment}-vnet",
  "location": "$location",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

    log_success "Azure infrastructure provisioned for $environment"
}

# GCP provisioning functions
provision_gcp() {
    local environment=$1
    local project=$2
    local region=${3:-us-central1}

    log_info "Provisioning GCP infrastructure in $region for $environment (project: $project)"

    # Set project
    gcloud config set project "$project"

    # Create VPC network
    gcloud compute networks create "${environment}-vpc" \
        --subnet-mode=custom \
        --bgp-routing-mode=regional \
        > /dev/null

    log_info "Created VPC network: ${environment}-vpc"

    # Create subnet
    gcloud compute networks subnets create "${environment}-subnet" \
        --network="${environment}-vpc" \
        --region="$region" \
        --range="10.2.0.0/24" \
        > /dev/null

    log_info "Created subnet: ${environment}-subnet"

    # Save state
    cat > "$STATE_DIR/gcp/${environment}.json" <<EOF
{
  "project": "$project",
  "vpc": "${environment}-vpc",
  "subnet": "${environment}-subnet",
  "region": "$region",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

    log_success "GCP infrastructure provisioned for $environment"
}

# Terraform-based provisioning
provision_with_terraform() {
    local config_dir=$1
    local environment=$2

    log_info "Provisioning infrastructure with Terraform from $config_dir"

    cd "$config_dir" || exit 1

    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init -upgrade

    # Validate configuration
    log_info "Validating Terraform configuration..."
    terraform validate

    # Plan infrastructure changes
    log_info "Planning infrastructure changes..."
    terraform plan \
        -var="environment=$environment" \
        -out="$STATE_DIR/terraform/${environment}.tfplan"

    # Apply changes
    read -rp "Apply Terraform changes? (yes/no): " confirm
    if [[ $confirm == "yes" ]]; then
        log_info "Applying Terraform changes..."
        terraform apply "$STATE_DIR/terraform/${environment}.tfplan"

        # Save Terraform state
        terraform output -json > "$STATE_DIR/terraform/${environment}.json"

        log_success "Terraform infrastructure provisioned"
    else
        log_warning "Terraform apply cancelled"
    fi
}

# Health check function
health_check() {
    local environment=$1

    log_info "Running health checks for $environment"

    local checks_passed=0
    local checks_total=0

    # Check AWS resources
    if [ -f "$STATE_DIR/aws/${environment}.json" ]; then
        ((checks_total++))
        local vpc_id
        vpc_id=$(jq -r '.vpc_id' "$STATE_DIR/aws/${environment}.json")
        if aws ec2 describe-vpcs --vpc-ids "$vpc_id" &> /dev/null; then
            ((checks_passed++))
            log_success "AWS VPC health check passed"
        else
            log_error "AWS VPC health check failed"
        fi
    fi

    # Check Azure resources
    if [ -f "$STATE_DIR/azure/${environment}.json" ]; then
        ((checks_total++))
        local resource_group
        resource_group=$(jq -r '.resource_group' "$STATE_DIR/azure/${environment}.json")
        if az group show --name "$resource_group" &> /dev/null; then
            ((checks_passed++))
            log_success "Azure resource group health check passed"
        else
            log_error "Azure resource group health check failed"
        fi
    fi

    # Check GCP resources
    if [ -f "$STATE_DIR/gcp/${environment}.json" ]; then
        ((checks_total++))
        local vpc project
        vpc=$(jq -r '.vpc' "$STATE_DIR/gcp/${environment}.json")
        project=$(jq -r '.project' "$STATE_DIR/gcp/${environment}.json")
        if gcloud compute networks describe "$vpc" --project="$project" &> /dev/null; then
            ((checks_passed++))
            log_success "GCP VPC health check passed"
        else
            log_error "GCP VPC health check failed"
        fi
    fi

    log_info "Health checks: $checks_passed/$checks_total passed"

    if [ "$checks_passed" -eq "$checks_total" ]; then
        return 0
    else
        return 1
    fi
}

# Cleanup function
cleanup() {
    local environment=$1
    local provider=${2:-all}

    log_warning "Cleaning up infrastructure for $environment (provider: $provider)"

    read -rp "This will destroy all resources. Are you sure? (yes/no): " confirm
    if [[ $confirm != "yes" ]]; then
        log_info "Cleanup cancelled"
        return 0
    fi

    if [[ $provider == "aws" || $provider == "all" ]]; then
        if [ -f "$STATE_DIR/aws/${environment}.json" ]; then
            log_info "Cleaning up AWS resources..."
            local vpc_id
            vpc_id=$(jq -r '.vpc_id' "$STATE_DIR/aws/${environment}.json")
            aws ec2 delete-vpc --vpc-id "$vpc_id" || log_warning "Failed to delete AWS VPC"
            rm -f "$STATE_DIR/aws/${environment}.json"
        fi
    fi

    if [[ $provider == "azure" || $provider == "all" ]]; then
        if [ -f "$STATE_DIR/azure/${environment}.json" ]; then
            log_info "Cleaning up Azure resources..."
            local resource_group
            resource_group=$(jq -r '.resource_group' "$STATE_DIR/azure/${environment}.json")
            az group delete --name "$resource_group" --yes --no-wait || log_warning "Failed to delete Azure resource group"
            rm -f "$STATE_DIR/azure/${environment}.json"
        fi
    fi

    if [[ $provider == "gcp" || $provider == "all" ]]; then
        if [ -f "$STATE_DIR/gcp/${environment}.json" ]; then
            log_info "Cleaning up GCP resources..."
            local vpc project
            vpc=$(jq -r '.vpc' "$STATE_DIR/gcp/${environment}.json")
            project=$(jq -r '.project' "$STATE_DIR/gcp/${environment}.json")
            gcloud compute networks delete "$vpc" --project="$project" --quiet || log_warning "Failed to delete GCP VPC"
            rm -f "$STATE_DIR/gcp/${environment}.json"
        fi
    fi

    log_success "Cleanup completed"
}

# Main function
main() {
    local command=$1
    shift

    init_state_dir

    case $command in
        validate)
            validate_prerequisites
            ;;
        provision)
            local provider=$1
            local environment=$2
            shift 2

            validate_prerequisites

            case $provider in
                aws)
                    provision_aws "$environment" "$@"
                    ;;
                azure)
                    provision_azure "$environment" "$@"
                    ;;
                gcp)
                    provision_gcp "$environment" "$@"
                    ;;
                terraform)
                    provision_with_terraform "$@"
                    ;;
                *)
                    log_error "Unknown provider: $provider"
                    exit 1
                    ;;
            esac
            ;;
        health-check)
            health_check "$1"
            ;;
        cleanup)
            cleanup "$@"
            ;;
        *)
            echo "Usage: $0 {validate|provision|health-check|cleanup} [options]"
            echo ""
            echo "Commands:"
            echo "  validate                        - Validate prerequisites and credentials"
            echo "  provision aws <env> [region]    - Provision AWS infrastructure"
            echo "  provision azure <env> [location] - Provision Azure infrastructure"
            echo "  provision gcp <env> <project> [region] - Provision GCP infrastructure"
            echo "  provision terraform <dir> <env> - Provision with Terraform"
            echo "  health-check <env>              - Run health checks"
            echo "  cleanup <env> [provider]        - Clean up resources"
            exit 1
            ;;
    esac
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
