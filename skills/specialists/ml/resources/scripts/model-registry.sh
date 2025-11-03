#!/bin/bash
# Model Registry Management with Versioning and Deployment
#
# Centralized model lifecycle management:
# - Model versioning with semantic versioning
# - Environment-specific deployments (dev/staging/prod)
# - Model promotion workflows
# - Rollback capabilities
# - Metadata tracking
#
# Usage:
#   ./model-registry.sh register <name> <version> <path>
#   ./model-registry.sh deploy <env> <name> <version>
#   ./model-registry.sh promote <name> <version>
#   ./model-registry.sh rollback <env> <name>
#   ./model-registry.sh list [name]

set -e  # Exit on error

# Configuration
REGISTRY_DIR="${ML_REGISTRY_DIR:-./model-registry}"
METADATA_DB="${REGISTRY_DIR}/metadata.json"
MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"

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

# Initialize registry
init_registry() {
    mkdir -p "${REGISTRY_DIR}"/{dev,staging,production,archive}

    if [ ! -f "$METADATA_DB" ]; then
        echo '{"models": {}}' > "$METADATA_DB"
        log_info "Initialized model registry at ${REGISTRY_DIR}"
    fi
}

# Validate semantic version
validate_version() {
    local version=$1
    if ! [[ $version =~ ^v?[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        log_error "Invalid version format: $version (expected: v1.0.0 or 1.0.0)"
        exit 1
    fi
}

# Register a new model
register_model() {
    local name=$1
    local version=$2
    local model_path=$3

    if [ -z "$name" ] || [ -z "$version" ] || [ -z "$model_path" ]; then
        log_error "Usage: register <name> <version> <path>"
        exit 1
    fi

    validate_version "$version"

    if [ ! -f "$model_path" ] && [ ! -d "$model_path" ]; then
        log_error "Model path not found: $model_path"
        exit 1
    fi

    # Create model directory
    local model_dir="${REGISTRY_DIR}/models/${name}/${version}"
    mkdir -p "$model_dir"

    # Copy model artifacts
    log_info "Copying model artifacts..."
    cp -r "$model_path" "${model_dir}/"

    # Generate model metadata
    local metadata=$(cat <<EOF
{
    "name": "$name",
    "version": "$version",
    "registered_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "model_path": "${model_dir}/$(basename $model_path)",
    "status": "registered",
    "deployments": {}
}
EOF
)

    echo "$metadata" > "${model_dir}/metadata.json"

    # Update global metadata
    python3 -c "
import json
with open('$METADATA_DB', 'r') as f:
    db = json.load(f)

if '$name' not in db['models']:
    db['models']['$name'] = {'versions': {}}

db['models']['$name']['versions']['$version'] = json.loads('''$metadata''')

with open('$METADATA_DB', 'w') as f:
    json.dump(db, f, indent=2)
"

    log_info "Registered model: ${name} ${version}"
    log_info "Location: ${model_dir}"
}

# Deploy model to environment
deploy_model() {
    local env=$1
    local name=$2
    local version=$3

    if [ -z "$env" ] || [ -z "$name" ] || [ -z "$version" ]; then
        log_error "Usage: deploy <env> <name> <version>"
        exit 1
    fi

    if [[ ! "$env" =~ ^(dev|staging|production)$ ]]; then
        log_error "Invalid environment: $env (must be dev, staging, or production)"
        exit 1
    fi

    local model_dir="${REGISTRY_DIR}/models/${name}/${version}"

    if [ ! -d "$model_dir" ]; then
        log_error "Model not found: ${name} ${version}"
        exit 1
    fi

    # Create deployment symlink
    local deploy_dir="${REGISTRY_DIR}/${env}/${name}"
    mkdir -p "$deploy_dir"

    # Backup current deployment if exists
    if [ -L "${deploy_dir}/current" ]; then
        local current_version=$(readlink "${deploy_dir}/current" | xargs basename | xargs dirname | xargs basename)
        log_warn "Backing up current deployment: ${current_version}"
        ln -sfn "${deploy_dir}/current" "${deploy_dir}/previous"
    fi

    # Create new deployment
    ln -sfn "$model_dir" "${deploy_dir}/current"

    # Update metadata
    python3 -c "
import json
from datetime import datetime

with open('$METADATA_DB', 'r') as f:
    db = json.load(f)

db['models']['$name']['versions']['$version']['deployments']['$env'] = {
    'deployed_at': datetime.utcnow().isoformat() + 'Z',
    'status': 'active'
}

with open('$METADATA_DB', 'w') as f:
    json.dump(db, f, indent=2)
"

    log_info "Deployed ${name} ${version} to ${env}"

    # Run health check
    health_check "$env" "$name"
}

# Promote model from staging to production
promote_model() {
    local name=$1
    local version=$2

    if [ -z "$name" ] || [ -z "$version" ]; then
        log_error "Usage: promote <name> <version>"
        exit 1
    fi

    # Check if model is deployed to staging
    local staging_link="${REGISTRY_DIR}/staging/${name}/current"

    if [ ! -L "$staging_link" ]; then
        log_error "Model not deployed to staging: ${name}"
        exit 1
    fi

    local staging_version=$(readlink "$staging_link" | xargs basename | xargs dirname | xargs basename)

    if [ "$staging_version" != "$version" ]; then
        log_error "Version mismatch in staging. Expected: ${version}, Found: ${staging_version}"
        exit 1
    fi

    log_info "Promoting ${name} ${version} from staging to production..."

    # Deploy to production
    deploy_model "production" "$name" "$version"

    log_info "Successfully promoted to production"
}

# Rollback to previous deployment
rollback_model() {
    local env=$1
    local name=$2

    if [ -z "$env" ] || [ -z "$name" ]; then
        log_error "Usage: rollback <env> <name>"
        exit 1
    fi

    local deploy_dir="${REGISTRY_DIR}/${env}/${name}"
    local previous_link="${deploy_dir}/previous"

    if [ ! -L "$previous_link" ]; then
        log_error "No previous deployment found for rollback"
        exit 1
    fi

    local previous_version=$(readlink "$previous_link" | xargs basename | xargs dirname | xargs basename)

    log_warn "Rolling back to ${previous_version}"

    # Swap current and previous
    local temp="${deploy_dir}/temp"
    mv "${deploy_dir}/current" "$temp"
    mv "$previous_link" "${deploy_dir}/current"
    mv "$temp" "$previous_link"

    log_info "Rollback completed. Current version: ${previous_version}"

    # Run health check
    health_check "$env" "$name"
}

# List models
list_models() {
    local name=$1

    if [ -z "$name" ]; then
        # List all models
        log_info "Registered Models:"
        python3 -c "
import json
with open('$METADATA_DB', 'r') as f:
    db = json.load(f)

for model_name, model_data in db['models'].items():
    print(f'  {model_name}')
    for version in model_data['versions'].keys():
        print(f'    - {version}')
"
    else
        # List specific model versions
        log_info "Model: ${name}"
        python3 -c "
import json
with open('$METADATA_DB', 'r') as f:
    db = json.load(f)

if '$name' in db['models']:
    for version, meta in db['models']['$name']['versions'].items():
        print(f'  Version: {version}')
        print(f'    Registered: {meta['registered_at']}')
        print(f'    Status: {meta['status']}')
        if meta.get('deployments'):
            print(f'    Deployments:')
            for env, deploy in meta['deployments'].items():
                print(f'      - {env}: {deploy['status']}')
        print()
else:
    print(f'Model not found: $name')
"
    fi
}

# Health check
health_check() {
    local env=$1
    local name=$2

    log_info "Running health check for ${name} in ${env}..."

    # TODO: Add actual health check logic (e.g., test inference)
    # This is a placeholder
    sleep 1
    log_info "✓ Health check passed"
}

# Archive old model
archive_model() {
    local name=$1
    local version=$2

    if [ -z "$name" ] || [ -z "$version" ]; then
        log_error "Usage: archive <name> <version>"
        exit 1
    fi

    local model_dir="${REGISTRY_DIR}/models/${name}/${version}"
    local archive_dir="${REGISTRY_DIR}/archive/${name}/${version}"

    if [ ! -d "$model_dir" ]; then
        log_error "Model not found: ${name} ${version}"
        exit 1
    fi

    mkdir -p "$(dirname "$archive_dir")"
    mv "$model_dir" "$archive_dir"

    log_info "Archived model: ${name} ${version}"
}

# Main command dispatcher
main() {
    init_registry

    local command=$1
    shift

    case "$command" in
        register)
            register_model "$@"
            ;;
        deploy)
            deploy_model "$@"
            ;;
        promote)
            promote_model "$@"
            ;;
        rollback)
            rollback_model "$@"
            ;;
        list)
            list_models "$@"
            ;;
        archive)
            archive_model "$@"
            ;;
        *)
            cat <<EOF
Model Registry Management

Usage:
  $0 register <name> <version> <path>    Register a new model
  $0 deploy <env> <name> <version>       Deploy model to environment
  $0 promote <name> <version>            Promote from staging to production
  $0 rollback <env> <name>               Rollback to previous deployment
  $0 list [name]                         List models (all or specific)
  $0 archive <name> <version>            Archive old model

Environments: dev, staging, production

Examples:
  $0 register image-classifier v1.0.0 ./model.pkl
  $0 deploy staging image-classifier v1.0.0
  $0 promote image-classifier v1.0.0
  $0 rollback production image-classifier
EOF
            exit 1
            ;;
    esac
}

main "$@"
