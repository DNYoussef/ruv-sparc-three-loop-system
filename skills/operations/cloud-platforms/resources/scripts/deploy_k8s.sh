#!/bin/bash
#
# Kubernetes Deployment Automation Script
# Handles Helm charts, kubectl deployments, and cluster operations
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check required tools
check_dependencies() {
    local missing=()

    command -v kubectl >/dev/null 2>&1 || missing+=("kubectl")
    command -v helm >/dev/null 2>&1 || missing+=("helm")
    command -v jq >/dev/null 2>&1 || missing+=("jq")

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        log_error "Please install: https://kubernetes.io/docs/tasks/tools/"
        exit 1
    fi
}

# Deploy with kubectl
deploy_kubectl() {
    local manifest="$1"
    local namespace="${2:-default}"

    log_info "Deploying manifest: $manifest to namespace: $namespace"

    # Create namespace if not exists
    if ! kubectl get namespace "$namespace" &>/dev/null; then
        log_info "Creating namespace: $namespace"
        kubectl create namespace "$namespace"
    fi

    # Apply manifest
    kubectl apply -f "$manifest" -n "$namespace"

    # Wait for deployment to be ready
    if grep -q "kind: Deployment" "$manifest"; then
        local deployment_name
        deployment_name=$(grep "name:" "$manifest" | head -1 | awk '{print $2}')
        log_info "Waiting for deployment: $deployment_name"
        kubectl rollout status deployment/"$deployment_name" -n "$namespace" --timeout=300s
    fi

    log_info "Deployment successful"
}

# Deploy Helm chart
deploy_helm() {
    local release_name="$1"
    local chart_path="$2"
    local namespace="${3:-default}"
    local values_file="${4:-}"

    log_info "Deploying Helm chart: $chart_path as release: $release_name"

    # Create namespace if not exists
    if ! kubectl get namespace "$namespace" &>/dev/null; then
        log_info "Creating namespace: $namespace"
        kubectl create namespace "$namespace"
    fi

    # Build helm command
    local helm_cmd="helm upgrade --install $release_name $chart_path --namespace $namespace --create-namespace"

    if [ -n "$values_file" ]; then
        helm_cmd="$helm_cmd --values $values_file"
    fi

    # Execute helm deployment
    eval "$helm_cmd"

    # Wait for deployment
    log_info "Waiting for release: $release_name"
    kubectl wait --for=condition=available --timeout=300s \
        deployment -l "app.kubernetes.io/instance=$release_name" -n "$namespace" 2>/dev/null || true

    log_info "Helm deployment successful"
    helm status "$release_name" -n "$namespace"
}

# Create deployment from Docker image
deploy_image() {
    local name="$1"
    local image="$2"
    local namespace="${3:-default}"
    local replicas="${4:-1}"
    local port="${5:-80}"

    log_info "Creating deployment: $name from image: $image"

    # Create namespace if not exists
    if ! kubectl get namespace "$namespace" &>/dev/null; then
        kubectl create namespace "$namespace"
    fi

    # Create deployment
    kubectl create deployment "$name" \
        --image="$image" \
        --replicas="$replicas" \
        --namespace="$namespace" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Expose deployment
    kubectl expose deployment "$name" \
        --port="$port" \
        --target-port="$port" \
        --namespace="$namespace" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Wait for rollout
    kubectl rollout status deployment/"$name" -n "$namespace" --timeout=300s

    log_info "Image deployment successful"
}

# Scale deployment
scale_deployment() {
    local name="$1"
    local replicas="$2"
    local namespace="${3:-default}"

    log_info "Scaling deployment: $name to $replicas replicas"

    kubectl scale deployment/"$name" --replicas="$replicas" -n "$namespace"
    kubectl rollout status deployment/"$name" -n "$namespace" --timeout=300s

    log_info "Scaling successful"
}

# Rollback deployment
rollback_deployment() {
    local name="$1"
    local namespace="${2:-default}"
    local revision="${3:-0}"

    log_info "Rolling back deployment: $name"

    if [ "$revision" -eq 0 ]; then
        kubectl rollout undo deployment/"$name" -n "$namespace"
    else
        kubectl rollout undo deployment/"$name" --to-revision="$revision" -n "$namespace"
    fi

    kubectl rollout status deployment/"$name" -n "$namespace" --timeout=300s

    log_info "Rollback successful"
}

# Get deployment status
get_status() {
    local name="$1"
    local namespace="${2:-default}"

    log_info "Deployment status for: $name"

    kubectl get deployment "$name" -n "$namespace"
    echo ""
    kubectl get pods -l "app=$name" -n "$namespace"
    echo ""
    kubectl get services -l "app=$name" -n "$namespace"
}

# Create ConfigMap from file
create_configmap() {
    local name="$1"
    local file="$2"
    local namespace="${3:-default}"

    log_info "Creating ConfigMap: $name from file: $file"

    kubectl create configmap "$name" \
        --from-file="$file" \
        --namespace="$namespace" \
        --dry-run=client -o yaml | kubectl apply -f -

    log_info "ConfigMap created successfully"
}

# Create Secret from file
create_secret() {
    local name="$1"
    local file="$2"
    local namespace="${3:-default}"

    log_info "Creating Secret: $name from file: $file"

    kubectl create secret generic "$name" \
        --from-file="$file" \
        --namespace="$namespace" \
        --dry-run=client -o yaml | kubectl apply -f -

    log_info "Secret created successfully"
}

# Main CLI
main() {
    check_dependencies

    local command="${1:-help}"
    shift || true

    case "$command" in
        kubectl)
            [ $# -lt 1 ] && { log_error "Usage: $0 kubectl <manifest> [namespace]"; exit 1; }
            deploy_kubectl "$@"
            ;;
        helm)
            [ $# -lt 2 ] && { log_error "Usage: $0 helm <release> <chart> [namespace] [values-file]"; exit 1; }
            deploy_helm "$@"
            ;;
        image)
            [ $# -lt 2 ] && { log_error "Usage: $0 image <name> <image> [namespace] [replicas] [port]"; exit 1; }
            deploy_image "$@"
            ;;
        scale)
            [ $# -lt 2 ] && { log_error "Usage: $0 scale <name> <replicas> [namespace]"; exit 1; }
            scale_deployment "$@"
            ;;
        rollback)
            [ $# -lt 1 ] && { log_error "Usage: $0 rollback <name> [namespace] [revision]"; exit 1; }
            rollback_deployment "$@"
            ;;
        status)
            [ $# -lt 1 ] && { log_error "Usage: $0 status <name> [namespace]"; exit 1; }
            get_status "$@"
            ;;
        configmap)
            [ $# -lt 2 ] && { log_error "Usage: $0 configmap <name> <file> [namespace]"; exit 1; }
            create_configmap "$@"
            ;;
        secret)
            [ $# -lt 2 ] && { log_error "Usage: $0 secret <name> <file> [namespace]"; exit 1; }
            create_secret "$@"
            ;;
        help|*)
            cat <<EOF
Kubernetes Deployment Automation Script

Usage: $0 <command> [options]

Commands:
  kubectl <manifest> [namespace]                  - Deploy using kubectl
  helm <release> <chart> [namespace] [values]    - Deploy Helm chart
  image <name> <image> [namespace] [replicas]    - Deploy from Docker image
  scale <name> <replicas> [namespace]            - Scale deployment
  rollback <name> [namespace] [revision]         - Rollback deployment
  status <name> [namespace]                      - Get deployment status
  configmap <name> <file> [namespace]            - Create ConfigMap
  secret <name> <file> [namespace]               - Create Secret
  help                                           - Show this help

Examples:
  $0 kubectl deployment.yaml production
  $0 helm myapp ./charts/myapp production values.yaml
  $0 image webapp nginx:latest production 3 80
  $0 scale webapp 5 production
  $0 rollback webapp production
  $0 status webapp production

EOF
            ;;
    esac
}

main "$@"
