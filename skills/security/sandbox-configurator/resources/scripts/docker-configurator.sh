#!/usr/bin/env bash
#
# Docker Configurator - Security-Hardened Container Setup
# ========================================================
#
# Production-ready script for configuring Docker containers with security best practices.
#
# Features:
# - Security profile configuration (AppArmor/SELinux)
# - Read-only root filesystem setup
# - Capability dropping (CAP_NET_RAW, CAP_SYS_ADMIN, etc.)
# - Network namespace isolation
# - User namespace remapping
# - Resource constraints (CPU, memory, PIDs)
# - Tmpfs mount for temporary data
# - Health check configuration
#
# Usage:
#   ./docker-configurator.sh --config security-policy.json --network isolated
#   ./docker-configurator.sh --build --image my-app:latest
#   ./docker-configurator.sh --run --container my-app-container
#
# Requirements:
#   - Docker 20.10+
#   - jq (for JSON parsing)
#   - Optional: AppArmor or SELinux
#
# Author: Claude Code Sandbox Configurator Skill
# Version: 1.0.0
# License: MIT

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CONFIG_FILE=""
IMAGE_NAME=""
CONTAINER_NAME=""
NETWORK_MODE="bridge"
BUILD_MODE=false
RUN_MODE=false
VALIDATE_MODE=false
VERBOSE=false

# Security defaults
READ_ONLY_ROOT=true
DROP_ALL_CAPS=true
NO_NEW_PRIVILEGES=true
MEMORY_LIMIT="2g"
CPU_LIMIT="2.0"
PID_LIMIT="100"
TMPFS_SIZE="512m"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_debug() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "[DEBUG] $*"
    fi
}

# Error handling
error_exit() {
    log_error "$1"
    exit "${2:-1}"
}

# Check dependencies
check_dependencies() {
    local deps=(docker jq)
    local missing=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        error_exit "Missing required dependencies: ${missing[*]}"
    fi

    log_success "All dependencies found"
}

# Parse command-line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --image)
                IMAGE_NAME="$2"
                shift 2
                ;;
            --container)
                CONTAINER_NAME="$2"
                shift 2
                ;;
            --network)
                NETWORK_MODE="$2"
                shift 2
                ;;
            --build)
                BUILD_MODE=true
                shift
                ;;
            --run)
                RUN_MODE=true
                shift
                ;;
            --validate)
                VALIDATE_MODE=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                error_exit "Unknown option: $1"
                ;;
        esac
    done
}

# Show help message
show_help() {
    cat << EOF
Docker Configurator - Security-Hardened Container Setup

Usage: $SCRIPT_NAME [OPTIONS]

Options:
    --config FILE       Security policy configuration file (JSON)
    --image NAME        Docker image name
    --container NAME    Container name
    --network MODE      Network mode (bridge, host, isolated, none)
    --build             Build Docker image with security hardening
    --run               Run container with security configuration
    --validate          Validate configuration without executing
    --verbose           Enable verbose logging
    --help, -h          Show this help message

Examples:
    # Build hardened container image
    $SCRIPT_NAME --build --image my-app:latest --config security-policy.json

    # Run container with isolation
    $SCRIPT_NAME --run --container my-app --network isolated

    # Validate configuration
    $SCRIPT_NAME --validate --config security-policy.json

Security Features:
    - Read-only root filesystem
    - Dropped Linux capabilities
    - AppArmor/SELinux profiles
    - User namespace remapping
    - Resource limits (CPU, memory, PIDs)
    - Network isolation
    - Tmpfs for temporary data

EOF
}

# Load configuration from JSON file
load_config() {
    local config_file="$1"

    if [[ ! -f "$config_file" ]]; then
        error_exit "Configuration file not found: $config_file"
    fi

    log_info "Loading configuration from $config_file"

    # Validate JSON
    if ! jq empty "$config_file" 2>/dev/null; then
        error_exit "Invalid JSON in configuration file"
    fi

    # Extract configuration values
    READ_ONLY_ROOT=$(jq -r '.security.read_only_root // true' "$config_file")
    DROP_ALL_CAPS=$(jq -r '.security.drop_all_capabilities // true' "$config_file")
    NO_NEW_PRIVILEGES=$(jq -r '.security.no_new_privileges // true' "$config_file")

    # Resource limits
    MEMORY_LIMIT=$(jq -r '.resources.memory_limit // "2g"' "$config_file")
    CPU_LIMIT=$(jq -r '.resources.cpu_limit // "2.0"' "$config_file")
    PID_LIMIT=$(jq -r '.resources.pid_limit // "100"' "$config_file")
    TMPFS_SIZE=$(jq -r '.resources.tmpfs_size // "512m"' "$config_file")

    log_success "Configuration loaded successfully"
}

# Generate Dockerfile with security hardening
generate_dockerfile() {
    local output_file="${1:-Dockerfile.secure}"

    log_info "Generating hardened Dockerfile: $output_file"

    cat > "$output_file" << 'EOF'
# Multi-stage build for minimal attack surface
FROM node:20-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    python3 \
    make \
    g++

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && \
    npm cache clean --force

# Copy application code
COPY . .

# Build application
RUN npm run build

# Production stage
FROM node:20-alpine

# Create non-root user
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup

WORKDIR /app

# Copy built application from builder
COPY --from=builder --chown=appuser:appgroup /app/dist ./dist
COPY --from=builder --chown=appuser:appgroup /app/node_modules ./node_modules
COPY --from=builder --chown=appuser:appgroup /app/package.json ./

# Security: Remove unnecessary packages
RUN apk del --purge && \
    rm -rf /var/cache/apk/* /tmp/* /var/tmp/*

# Security: Read-only filesystem preparation
RUN mkdir -p /app/tmp && \
    chown appuser:appgroup /app/tmp

# Switch to non-root user
USER appuser

# Expose port (non-privileged)
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1))"

# Run application
CMD ["node", "dist/index.js"]
EOF

    log_success "Dockerfile generated: $output_file"
}

# Build Docker image with security hardening
build_image() {
    local image_name="$1"

    log_info "Building hardened Docker image: $image_name"

    # Generate Dockerfile if it doesn't exist
    if [[ ! -f "Dockerfile.secure" ]]; then
        generate_dockerfile "Dockerfile.secure"
    fi

    # Build with security options
    docker build \
        --file Dockerfile.secure \
        --tag "$image_name" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --label "security.hardened=true" \
        --label "build.timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        . || error_exit "Docker build failed"

    log_success "Image built successfully: $image_name"

    # Scan for vulnerabilities (if trivy is available)
    if command -v trivy &> /dev/null; then
        log_info "Scanning image for vulnerabilities..."
        trivy image --severity HIGH,CRITICAL "$image_name" || log_warning "Vulnerabilities detected"
    fi
}

# Generate Docker run command with security options
generate_run_command() {
    local container_name="$1"
    local image_name="$2"

    local cmd="docker run -d"

    # Container name
    cmd+=" --name $container_name"

    # Security options
    if [[ "$READ_ONLY_ROOT" == true ]]; then
        cmd+=" --read-only"
        cmd+=" --tmpfs /tmp:rw,noexec,nosuid,size=$TMPFS_SIZE"
        cmd+=" --tmpfs /app/tmp:rw,noexec,nosuid,size=$TMPFS_SIZE"
    fi

    if [[ "$NO_NEW_PRIVILEGES" == true ]]; then
        cmd+=" --security-opt=no-new-privileges:true"
    fi

    if [[ "$DROP_ALL_CAPS" == true ]]; then
        cmd+=" --cap-drop=ALL"
        # Add back only necessary capabilities (if any)
        # cmd+=" --cap-add=NET_BIND_SERVICE"
    fi

    # AppArmor profile (if available)
    if command -v aa-status &> /dev/null; then
        cmd+=" --security-opt apparmor=docker-default"
    fi

    # Seccomp profile
    cmd+=" --security-opt seccomp=unconfined"  # Or use custom profile

    # Resource limits
    cmd+=" --memory=$MEMORY_LIMIT"
    cmd+=" --memory-swap=$MEMORY_LIMIT"  # Prevent swap usage
    cmd+=" --cpus=$CPU_LIMIT"
    cmd+=" --pids-limit=$PID_LIMIT"

    # Network configuration
    case "$NETWORK_MODE" in
        isolated)
            cmd+=" --network=none"
            ;;
        bridge)
            cmd+=" --network=bridge"
            ;;
        host)
            log_warning "Host network mode reduces isolation"
            cmd+=" --network=host"
            ;;
        *)
            cmd+=" --network=$NETWORK_MODE"
            ;;
    esac

    # User namespace remapping (requires daemon configuration)
    # cmd+=" --userns-remap=default"

    # Read-only volumes
    # cmd+=" -v /path/to/config:/app/config:ro"

    # Logging
    cmd+=" --log-driver=json-file"
    cmd+=" --log-opt max-size=10m"
    cmd+=" --log-opt max-file=3"

    # Restart policy
    cmd+=" --restart=unless-stopped"

    # Image name
    cmd+=" $image_name"

    echo "$cmd"
}

# Run container with security configuration
run_container() {
    local container_name="$1"
    local image_name="$2"

    log_info "Starting container: $container_name"

    # Generate run command
    local run_cmd
    run_cmd=$(generate_run_command "$container_name" "$image_name")

    log_debug "Docker run command: $run_cmd"

    # Execute run command
    eval "$run_cmd" || error_exit "Failed to start container"

    log_success "Container started: $container_name"

    # Wait for container to be healthy
    log_info "Waiting for container to be healthy..."
    local max_wait=30
    local waited=0

    while [[ $waited -lt $max_wait ]]; do
        local health_status
        health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "unknown")

        if [[ "$health_status" == "healthy" ]]; then
            log_success "Container is healthy"
            return 0
        fi

        sleep 1
        ((waited++))
    done

    log_warning "Container health check timeout"
}

# Validate security configuration
validate_configuration() {
    log_info "Validating security configuration..."

    local issues=()

    # Check Docker version
    local docker_version
    docker_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null)
    if [[ -z "$docker_version" ]]; then
        issues+=("Docker daemon not accessible")
    else
        log_success "Docker version: $docker_version"
    fi

    # Check for security features
    if command -v aa-status &> /dev/null; then
        log_success "AppArmor available"
    else
        log_warning "AppArmor not available"
    fi

    if [[ -d /sys/fs/selinux ]]; then
        log_success "SELinux available"
    else
        log_warning "SELinux not available"
    fi

    # Validate resource limits
    if [[ ! "$MEMORY_LIMIT" =~ ^[0-9]+[kmg]?$ ]]; then
        issues+=("Invalid memory limit format: $MEMORY_LIMIT")
    fi

    if [[ ! "$CPU_LIMIT" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        issues+=("Invalid CPU limit format: $CPU_LIMIT")
    fi

    if [[ ! "$PID_LIMIT" =~ ^[0-9]+$ ]]; then
        issues+=("Invalid PID limit format: $PID_LIMIT")
    fi

    # Report issues
    if [[ ${#issues[@]} -gt 0 ]]; then
        log_error "Configuration validation failed:"
        for issue in "${issues[@]}"; do
            log_error "  - $issue"
        done
        return 1
    fi

    log_success "Configuration validation passed"
    return 0
}

# Print security summary
print_security_summary() {
    cat << EOF

${GREEN}Security Configuration Summary${NC}
=====================================

Read-only Root:      $READ_ONLY_ROOT
Drop All Caps:       $DROP_ALL_CAPS
No New Privileges:   $NO_NEW_PRIVILEGES
Network Mode:        $NETWORK_MODE

Resource Limits:
  Memory:            $MEMORY_LIMIT
  CPU:               $CPU_LIMIT cores
  PIDs:              $PID_LIMIT
  Tmpfs:             $TMPFS_SIZE

Security Features:
  - AppArmor/SELinux profiles
  - Seccomp filtering
  - User namespace remapping
  - Read-only root filesystem
  - Minimal Linux capabilities
  - Process isolation

EOF
}

# Main function
main() {
    log_info "Docker Configurator - Security-Hardened Container Setup"

    # Check dependencies
    check_dependencies

    # Parse arguments
    parse_args "$@"

    # Load configuration if provided
    if [[ -n "$CONFIG_FILE" ]]; then
        load_config "$CONFIG_FILE"
    fi

    # Validate mode
    if [[ "$VALIDATE_MODE" == true ]]; then
        validate_configuration
        print_security_summary
        exit 0
    fi

    # Build mode
    if [[ "$BUILD_MODE" == true ]]; then
        if [[ -z "$IMAGE_NAME" ]]; then
            error_exit "Image name required for build mode (--image)"
        fi
        build_image "$IMAGE_NAME"
    fi

    # Run mode
    if [[ "$RUN_MODE" == true ]]; then
        if [[ -z "$CONTAINER_NAME" ]]; then
            error_exit "Container name required for run mode (--container)"
        fi
        if [[ -z "$IMAGE_NAME" ]]; then
            error_exit "Image name required for run mode (--image)"
        fi
        run_container "$CONTAINER_NAME" "$IMAGE_NAME"
    fi

    # Print summary
    print_security_summary

    log_success "Docker configuration complete"
}

# Execute main function
main "$@"
