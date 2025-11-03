#!/bin/bash
#
# Google Cloud Platform Deployment Script
# Handles Cloud Run, Cloud Functions, GKE, and Compute Engine deployments
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Check gcloud CLI
check_gcloud() {
    if ! command -v gcloud &>/dev/null; then
        log_error "gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Not authenticated. Run: gcloud auth login"
        exit 1
    fi
}

# Set GCP project
set_project() {
    local project_id="$1"
    log_info "Setting project: $project_id"
    gcloud config set project "$project_id"
}

# Deploy to Cloud Run
deploy_cloud_run() {
    local service_name="$1"
    local image="$2"
    local region="${3:-us-central1}"
    local port="${4:-8080}"
    local memory="${5:-512Mi}"
    local cpu="${6:-1}"

    log_step "Deploying Cloud Run service: $service_name"

    gcloud run deploy "$service_name" \
        --image="$image" \
        --platform=managed \
        --region="$region" \
        --port="$port" \
        --memory="$memory" \
        --cpu="$cpu" \
        --allow-unauthenticated \
        --quiet

    # Get service URL
    local service_url
    service_url=$(gcloud run services describe "$service_name" \
        --region="$region" \
        --format="value(status.url)")

    log_info "Cloud Run service deployed: $service_url"
    echo "$service_url"
}

# Deploy Cloud Function (Gen 2)
deploy_cloud_function() {
    local function_name="$1"
    local source_dir="$2"
    local entry_point="$3"
    local runtime="${4:-python311}"
    local region="${5:-us-central1}"
    local memory="${6:-256MB}"
    local trigger="${7:-http}"

    log_step "Deploying Cloud Function: $function_name"

    local deploy_cmd="gcloud functions deploy $function_name \
        --gen2 \
        --runtime=$runtime \
        --region=$region \
        --source=$source_dir \
        --entry-point=$entry_point \
        --memory=$memory"

    if [ "$trigger" = "http" ]; then
        deploy_cmd="$deploy_cmd --trigger-http --allow-unauthenticated"
    else
        deploy_cmd="$deploy_cmd --trigger-topic=$trigger"
    fi

    eval "$deploy_cmd --quiet"

    # Get function URL
    if [ "$trigger" = "http" ]; then
        local function_url
        function_url=$(gcloud functions describe "$function_name" \
            --gen2 \
            --region="$region" \
            --format="value(serviceConfig.uri)")

        log_info "Cloud Function deployed: $function_url"
        echo "$function_url"
    else
        log_info "Cloud Function deployed (Pub/Sub trigger: $trigger)"
    fi
}

# Deploy to GKE
deploy_gke() {
    local cluster_name="$1"
    local zone="$2"
    local manifest="$3"
    local namespace="${4:-default}"

    log_step "Deploying to GKE cluster: $cluster_name"

    # Get cluster credentials
    log_info "Getting cluster credentials..."
    gcloud container clusters get-credentials "$cluster_name" --zone="$zone"

    # Create namespace if not exists
    if ! kubectl get namespace "$namespace" &>/dev/null; then
        log_info "Creating namespace: $namespace"
        kubectl create namespace "$namespace"
    fi

    # Apply manifest
    log_info "Applying manifest..."
    kubectl apply -f "$manifest" -n "$namespace"

    # Wait for deployment
    if grep -q "kind: Deployment" "$manifest"; then
        local deployment_name
        deployment_name=$(grep "name:" "$manifest" | head -1 | awk '{print $2}')
        log_info "Waiting for deployment: $deployment_name"
        kubectl rollout status deployment/"$deployment_name" -n "$namespace" --timeout=300s
    fi

    log_info "GKE deployment successful"
}

# Create GKE cluster
create_gke_cluster() {
    local cluster_name="$1"
    local zone="$2"
    local machine_type="${3:-e2-medium}"
    local num_nodes="${4:-3}"
    local enable_autoscaling="${5:-true}"

    log_step "Creating GKE cluster: $cluster_name"

    local create_cmd="gcloud container clusters create $cluster_name \
        --zone=$zone \
        --machine-type=$machine_type \
        --num-nodes=$num_nodes"

    if [ "$enable_autoscaling" = "true" ]; then
        create_cmd="$create_cmd --enable-autoscaling --min-nodes=1 --max-nodes=10"
    fi

    create_cmd="$create_cmd --enable-autorepair --enable-autoupgrade --quiet"

    eval "$create_cmd"

    log_info "GKE cluster created successfully"
}

# Deploy Compute Engine instance
deploy_compute_instance() {
    local instance_name="$1"
    local zone="$2"
    local machine_type="${3:-e2-medium}"
    local image_family="${4:-debian-11}"
    local image_project="${5:-debian-cloud}"
    local startup_script="${6:-}"

    log_step "Creating Compute Engine instance: $instance_name"

    local create_cmd="gcloud compute instances create $instance_name \
        --zone=$zone \
        --machine-type=$machine_type \
        --image-family=$image_family \
        --image-project=$image_project \
        --boot-disk-size=10GB \
        --boot-disk-type=pd-standard"

    if [ -n "$startup_script" ]; then
        create_cmd="$create_cmd --metadata-from-file=startup-script=$startup_script"
    fi

    create_cmd="$create_cmd --quiet"

    eval "$create_cmd"

    # Get instance IP
    local external_ip
    external_ip=$(gcloud compute instances describe "$instance_name" \
        --zone="$zone" \
        --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

    log_info "Compute Engine instance created: $external_ip"
    echo "$external_ip"
}

# Build and push Docker image to GCR
build_and_push_image() {
    local image_name="$1"
    local dockerfile_path="${2:-.}"
    local project_id
    project_id=$(gcloud config get-value project)

    log_step "Building and pushing image: $image_name"

    # Build with Cloud Build
    gcloud builds submit \
        --tag="gcr.io/$project_id/$image_name" \
        "$dockerfile_path" \
        --quiet

    local image_url="gcr.io/$project_id/$image_name"
    log_info "Image pushed: $image_url"
    echo "$image_url"
}

# Deploy App Engine
deploy_app_engine() {
    local app_yaml="$1"
    local version="${2:-}"

    log_step "Deploying to App Engine"

    local deploy_cmd="gcloud app deploy $app_yaml --quiet"

    if [ -n "$version" ]; then
        deploy_cmd="$deploy_cmd --version=$version"
    fi

    eval "$deploy_cmd"

    # Get app URL
    local app_url
    app_url=$(gcloud app describe --format="value(defaultHostname)")

    log_info "App Engine deployed: https://$app_url"
    echo "https://$app_url"
}

# Main CLI
main() {
    check_gcloud

    local command="${1:-help}"
    shift || true

    case "$command" in
        cloud-run)
            [ $# -lt 2 ] && { log_error "Usage: $0 cloud-run <service> <image> [region] [port] [memory] [cpu]"; exit 1; }
            deploy_cloud_run "$@"
            ;;
        function)
            [ $# -lt 3 ] && { log_error "Usage: $0 function <name> <source-dir> <entry-point> [runtime] [region] [memory] [trigger]"; exit 1; }
            deploy_cloud_function "$@"
            ;;
        gke)
            [ $# -lt 3 ] && { log_error "Usage: $0 gke <cluster> <zone> <manifest> [namespace]"; exit 1; }
            deploy_gke "$@"
            ;;
        create-gke)
            [ $# -lt 2 ] && { log_error "Usage: $0 create-gke <cluster> <zone> [machine-type] [num-nodes] [autoscale]"; exit 1; }
            create_gke_cluster "$@"
            ;;
        compute)
            [ $# -lt 2 ] && { log_error "Usage: $0 compute <instance> <zone> [machine-type] [image-family] [image-project] [startup-script]"; exit 1; }
            deploy_compute_instance "$@"
            ;;
        build)
            [ $# -lt 1 ] && { log_error "Usage: $0 build <image-name> [dockerfile-path]"; exit 1; }
            build_and_push_image "$@"
            ;;
        app-engine)
            [ $# -lt 1 ] && { log_error "Usage: $0 app-engine <app.yaml> [version]"; exit 1; }
            deploy_app_engine "$@"
            ;;
        set-project)
            [ $# -lt 1 ] && { log_error "Usage: $0 set-project <project-id>"; exit 1; }
            set_project "$@"
            ;;
        help|*)
            cat <<EOF
Google Cloud Platform Deployment Script

Usage: $0 <command> [options]

Commands:
  cloud-run <service> <image> [region] [port] [memory] [cpu]
      Deploy container to Cloud Run

  function <name> <source> <entry-point> [runtime] [region] [memory] [trigger]
      Deploy Cloud Function (Gen 2)

  gke <cluster> <zone> <manifest> [namespace]
      Deploy to existing GKE cluster

  create-gke <cluster> <zone> [machine-type] [nodes] [autoscale]
      Create new GKE cluster

  compute <instance> <zone> [machine-type] [image-family] [image-project] [startup-script]
      Create Compute Engine instance

  build <image-name> [dockerfile-path]
      Build and push Docker image to GCR

  app-engine <app.yaml> [version]
      Deploy to App Engine

  set-project <project-id>
      Set GCP project

  help
      Show this help

Examples:
  $0 set-project my-project-id
  $0 cloud-run my-service gcr.io/my-project/my-image us-central1
  $0 function my-func ./src main python311 us-central1
  $0 gke my-cluster us-central1-a deployment.yaml production
  $0 create-gke my-cluster us-central1-a e2-medium 3 true
  $0 compute my-instance us-central1-a e2-medium debian-11
  $0 build my-app ./app
  $0 app-engine app.yaml v1

EOF
            ;;
    esac
}

main "$@"
