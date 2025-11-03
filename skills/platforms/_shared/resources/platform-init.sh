#!/bin/bash
# Platform Initialization Script
# Initializes Flow Nexus platform services with comprehensive setup

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PLATFORM_DIR="${PLATFORM_DIR:-platform}"
LOG_DIR="${LOG_DIR:-logs/platform}"
CONFIG_FILE="${CONFIG_FILE:-platform/config/flow-nexus.json}"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check Node.js
    if ! command -v node &> /dev/null; then
        error "Node.js is not installed. Please install Node.js 18+ first."
        exit 1
    fi

    local node_version=$(node -v | cut -d 'v' -f 2 | cut -d '.' -f 1)
    if [ "$node_version" -lt 18 ]; then
        error "Node.js version must be 18 or higher. Current: $(node -v)"
        exit 1
    fi

    # Check npm
    if ! command -v npm &> /dev/null; then
        error "npm is not installed."
        exit 1
    fi

    # Check Flow Nexus MCP
    if ! command -v npx &> /dev/null; then
        error "npx is not available."
        exit 1
    fi

    info "Prerequisites check passed ✓"
}

# Create directory structure
create_directories() {
    log "Creating directory structure..."

    mkdir -p "$PLATFORM_DIR"/{config,services,scripts,docs,templates}
    mkdir -p "$LOG_DIR"
    mkdir -p "$PLATFORM_DIR"/storage/{uploads,cache,temp}
    mkdir -p "$PLATFORM_DIR"/data/{database,backups}

    info "Directory structure created ✓"
}

# Initialize configuration
init_config() {
    log "Initializing platform configuration..."

    if [ -f "$CONFIG_FILE" ]; then
        warn "Configuration file already exists: $CONFIG_FILE"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            info "Skipping configuration initialization"
            return
        fi
    fi

    cat > "$CONFIG_FILE" << 'EOF'
{
  "platform": {
    "name": "Flow Nexus Platform",
    "version": "1.0.0",
    "environment": "development"
  },
  "authentication": {
    "type": "email_password",
    "session_timeout": 3600,
    "require_verification": true,
    "allow_registration": true
  },
  "services": {
    "sandboxes": {
      "enabled": true,
      "max_concurrent": 5,
      "default_timeout": 3600,
      "templates": ["node", "python", "react", "nextjs"]
    },
    "storage": {
      "enabled": true,
      "max_size_mb": 1000,
      "retention_days": 30,
      "buckets": ["platform-assets", "user-uploads", "backups"]
    },
    "databases": {
      "enabled": true,
      "max_connections": 10,
      "pool_size": 20,
      "connection_timeout": 5000
    },
    "workflows": {
      "enabled": true,
      "max_agents": 8,
      "queue_size": 100,
      "retry_attempts": 3
    },
    "real_time": {
      "enabled": true,
      "max_subscriptions": 50,
      "heartbeat_interval": 30000
    }
  },
  "limits": {
    "requests_per_minute": 60,
    "storage_mb": 1000,
    "compute_hours": 10,
    "bandwidth_gb": 5
  },
  "monitoring": {
    "enabled": true,
    "log_level": "info",
    "metrics_interval": 60000,
    "health_check_interval": 30000
  },
  "security": {
    "rate_limiting": true,
    "cors_enabled": true,
    "allowed_origins": ["http://localhost:3000"],
    "encryption_at_rest": true,
    "audit_logging": true
  }
}
EOF

    info "Configuration initialized: $CONFIG_FILE ✓"
}

# Check Flow Nexus MCP availability
check_flow_nexus() {
    log "Checking Flow Nexus MCP availability..."

    if npx flow-nexus@latest --version &> /dev/null; then
        local version=$(npx flow-nexus@latest --version)
        info "Flow Nexus MCP available: $version ✓"
    else
        error "Flow Nexus MCP not available"
        info "Install with: claude mcp add flow-nexus npx flow-nexus@latest mcp start"
        exit 1
    fi
}

# Initialize environment file
init_env() {
    log "Initializing environment configuration..."

    local env_file="$PLATFORM_DIR/.env"

    if [ -f "$env_file" ]; then
        warn "Environment file already exists: $env_file"
        return
    fi

    cat > "$env_file" << 'EOF'
# Platform Environment Configuration
# DO NOT commit this file to version control

# Platform
NODE_ENV=development
PORT=3000
LOG_LEVEL=info

# Flow Nexus
FLOW_NEXUS_API_URL=https://api.flow-nexus.ruv.io
FLOW_NEXUS_USER_ID=
FLOW_NEXUS_SESSION_TOKEN=

# Database
DATABASE_URL=postgresql://localhost:5432/platform_dev
DATABASE_POOL_SIZE=20

# Storage
STORAGE_BUCKET=platform-assets
STORAGE_MAX_SIZE_MB=1000

# Security
SESSION_SECRET=
JWT_SECRET=
ENCRYPTION_KEY=

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090

# External Services
ANTHROPIC_API_KEY=
E2B_API_KEY=
EOF

    info "Environment file created: $env_file ✓"
    warn "Please update environment variables in: $env_file"
}

# Install dependencies
install_dependencies() {
    log "Installing platform dependencies..."

    local package_json="$PLATFORM_DIR/package.json"

    if [ -f "$package_json" ]; then
        info "package.json already exists, skipping creation"
    else
        cat > "$package_json" << 'EOF'
{
  "name": "flow-nexus-platform",
  "version": "1.0.0",
  "description": "Flow Nexus platform services and infrastructure",
  "main": "index.js",
  "scripts": {
    "start": "node services/app.js",
    "dev": "nodemon services/app.js",
    "test": "jest",
    "lint": "eslint .",
    "init": "node scripts/init-services.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "dotenv": "^16.3.1",
    "axios": "^1.6.0",
    "pg": "^8.11.3",
    "redis": "^4.6.10"
  },
  "devDependencies": {
    "nodemon": "^3.0.1",
    "jest": "^29.7.0",
    "eslint": "^8.52.0"
  }
}
EOF
        info "package.json created ✓"
    fi

    cd "$PLATFORM_DIR"
    npm install
    cd - > /dev/null

    info "Dependencies installed ✓"
}

# Create health check endpoint
create_health_check() {
    log "Creating health check endpoint..."

    local health_script="$PLATFORM_DIR/scripts/health-check.sh"

    cat > "$health_script" << 'EOF'
#!/bin/bash
# Platform Health Check Script

check_service() {
    local service=$1
    local endpoint=$2

    if curl -s -f "$endpoint" > /dev/null 2>&1; then
        echo "✓ $service: healthy"
        return 0
    else
        echo "✗ $service: unhealthy"
        return 1
    fi
}

echo "Platform Health Check"
echo "===================="

check_service "Application" "http://localhost:3000/health"
check_service "Metrics" "http://localhost:9090/metrics"

echo "===================="
EOF

    chmod +x "$health_script"
    info "Health check script created: $health_script ✓"
}

# Generate summary
generate_summary() {
    log "Platform initialization complete!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Platform Initialization Summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Platform directory: $PLATFORM_DIR"
    echo "Configuration: $CONFIG_FILE"
    echo "Environment: $PLATFORM_DIR/.env"
    echo "Logs: $LOG_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Update environment variables: $PLATFORM_DIR/.env"
    echo "  2. Register Flow Nexus account: npx flow-nexus@latest register"
    echo "  3. Login to Flow Nexus: npx flow-nexus@latest login"
    echo "  4. Initialize services: npm run init"
    echo "  5. Start platform: npm start"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Main execution
main() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Flow Nexus Platform Initialization"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    check_prerequisites
    create_directories
    init_config
    check_flow_nexus
    init_env
    install_dependencies
    create_health_check
    generate_summary
}

# Run main function
main "$@"
