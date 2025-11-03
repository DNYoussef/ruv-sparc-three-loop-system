# Platform Setup Example
Complete end-to-end platform initialization and configuration

## Overview

This example demonstrates setting up a complete Flow Nexus platform instance from scratch, including:
- Initial platform configuration
- Service provisioning
- Security setup
- Monitoring configuration
- User authentication
- Application deployment

**Estimated Time**: 30-45 minutes
**Difficulty**: Intermediate
**Prerequisites**: Node.js 18+, npm, Flow Nexus MCP access

---

## Step 1: Platform Initialization

### 1.1 Run Platform Initialization Script

```bash
# Navigate to platform directory
cd C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\skills\platform

# Make init script executable (Linux/Mac)
chmod +x resources/platform-init.sh

# Run initialization
./resources/platform-init.sh
```

**Expected Output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Flow Nexus Platform Initialization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[2025-11-02 10:00:00] Checking prerequisites...
✓ Node.js version: 18.17.0
✓ npm version: 9.6.7
✓ Flow Nexus MCP available: 1.0.0

[2025-11-02 10:00:01] Creating directory structure...
✓ Directory structure created

[2025-11-02 10:00:02] Initializing platform configuration...
✓ Configuration initialized: platform/config/flow-nexus.json

[2025-11-02 10:00:03] Platform initialization complete!
```

### 1.2 Verify Directory Structure

```bash
# List created directories
tree platform

platform/
├── config/
│   └── flow-nexus.json
├── services/
├── scripts/
│   └── health-check.sh
├── docs/
├── templates/
├── storage/
│   ├── uploads/
│   ├── cache/
│   └── temp/
└── data/
    ├── database/
    └── backups/
```

---

## Step 2: Configuration Setup

### 2.1 Update Environment Variables

```bash
# Edit environment file
nano platform/.env
```

```bash
# Platform Environment Configuration
NODE_ENV=development
PORT=3000
LOG_LEVEL=info

# Flow Nexus
FLOW_NEXUS_API_URL=https://api.flow-nexus.ruv.io
FLOW_NEXUS_USER_ID=<your-user-id>
FLOW_NEXUS_SESSION_TOKEN=<your-session-token>

# Database
DATABASE_URL=postgresql://localhost:5432/platform_dev
DATABASE_POOL_SIZE=20
DATABASE_PASSWORD=<secure-password>

# Storage
STORAGE_BUCKET=platform-assets
STORAGE_MAX_SIZE_MB=1000

# Security
SESSION_SECRET=<generate-random-secret>
JWT_SECRET=<generate-random-secret>
ENCRYPTION_KEY=<generate-random-key>

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090

# External Services
ANTHROPIC_API_KEY=<your-anthropic-key>
E2B_API_KEY=<your-e2b-key>
```

### 2.2 Customize Platform Configuration

```bash
# Edit platform config
nano platform/config/flow-nexus.json
```

```json
{
  "platform": {
    "name": "My Flow Nexus Platform",
    "version": "1.0.0",
    "environment": "development"
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
      "retention_days": 30
    },
    "databases": {
      "enabled": true,
      "max_connections": 10
    },
    "workflows": {
      "enabled": true,
      "max_agents": 8
    }
  },
  "limits": {
    "requests_per_minute": 60,
    "storage_mb": 1000,
    "compute_hours": 10
  }
}
```

---

## Step 3: Service Configuration

### 3.1 Create Service Manifest

```bash
# Copy service manifest template
cp resources/service-manifest.json platform/config/services.json

# Edit service configuration
nano platform/config/services.json
```

```json
{
  "version": "1.0.0",
  "services": [
    {
      "name": "api",
      "displayName": "API Server",
      "type": "web",
      "command": "node services/app.js",
      "port": 3000,
      "dependencies": ["database", "redis"],
      "healthCheck": {
        "type": "http",
        "endpoint": "http://localhost:3000/health",
        "interval": 30
      },
      "resources": {
        "cpu": "1",
        "memory": "512Mi"
      }
    },
    {
      "name": "database",
      "displayName": "PostgreSQL Database",
      "type": "database",
      "command": "pg_ctl start -D data/postgres",
      "port": 5432,
      "healthCheck": {
        "type": "tcp",
        "endpoint": "localhost:5432"
      }
    },
    {
      "name": "redis",
      "displayName": "Redis Cache",
      "type": "cache",
      "command": "redis-server --port 6379",
      "port": 6379,
      "healthCheck": {
        "type": "command",
        "command": "redis-cli ping"
      }
    }
  ]
}
```

### 3.2 Initialize Services

```bash
# Install dependencies
cd platform
npm install

# Initialize services
npm run init
```

**Expected Output:**
```
Initializing sandbox...
{ status: 'success', sandbox_id: 'sandbox_abc123' }

Initializing storage...
{ status: 'success', bucket: 'platform-assets' }

Initializing database...
{ status: 'success', connected: true }

All services initialized successfully
```

---

## Step 4: Authentication Setup

### 4.1 Register Flow Nexus Account

```bash
# Register new account
npx flow-nexus@latest register

# Follow prompts:
Email: user@example.com
Password: ********
Username: platform_user
Full Name: Platform User

# Verify email
# Check your email and click verification link
```

### 4.2 Login and Store Credentials

```bash
# Login to Flow Nexus
npx flow-nexus@latest login

Email: user@example.com
Password: ********

# Success! Session token saved
```

### 4.3 Update Environment with User ID

```bash
# Get user profile
npx flow-nexus@latest profile

# Copy user_id to .env
nano platform/.env

# Update:
FLOW_NEXUS_USER_ID=user_abc123xyz
```

---

## Step 5: Sandbox Creation

### 5.1 Create Development Sandbox

```javascript
// Create sandbox for Node.js development
const sandbox = await mcp__flow-nexus__sandbox_create({
  template: "node",
  name: "dev-sandbox",
  timeout: 3600,
  env_vars: {
    NODE_ENV: "development",
    LOG_LEVEL: "debug",
    DATABASE_URL: process.env.DATABASE_URL
  },
  install_packages: [
    "express",
    "axios",
    "dotenv",
    "pg"
  ]
});

console.log('Sandbox created:', sandbox.sandbox_id);
```

### 5.2 Configure Sandbox

```javascript
// Add additional configuration
await mcp__flow-nexus__sandbox_configure({
  sandbox_id: sandbox.sandbox_id,
  env_vars: {
    API_URL: "https://api.flow-nexus.ruv.io",
    MAX_RETRIES: "3"
  },
  install_packages: [
    "typescript",
    "jest",
    "eslint"
  ]
});
```

---

## Step 6: Storage Setup

### 6.1 Create Storage Buckets

```javascript
// Create platform assets bucket
await mcp__flow-nexus__storage_upload({
  bucket: "platform-assets",
  path: ".gitkeep",
  content: ""
});

// Create user uploads bucket
await mcp__flow-nexus__storage_upload({
  bucket: "user-uploads",
  path: ".gitkeep",
  content: ""
});

// Create backups bucket
await mcp__flow-nexus__storage_upload({
  bucket: "backups",
  path: ".gitkeep",
  content: ""
});
```

### 6.2 Upload Configuration Files

```javascript
// Upload platform configuration
const config = await fs.readFile('platform/config/flow-nexus.json', 'utf8');

await mcp__flow-nexus__storage_upload({
  bucket: "platform-assets",
  path: "config/flow-nexus.json",
  content: config,
  content_type: "application/json"
});
```

---

## Step 7: Database Setup

### 7.1 Create Database Schema

```sql
-- Create database schema
-- File: platform/config/schema.sql

CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  username VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  token VARCHAR(255) UNIQUE NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS applications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  name VARCHAR(255) NOT NULL,
  description TEXT,
  status VARCHAR(50) DEFAULT 'active',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS deployments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  application_id UUID REFERENCES applications(id) ON DELETE CASCADE,
  version VARCHAR(50) NOT NULL,
  sandbox_id VARCHAR(255),
  status VARCHAR(50) DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_sessions_token ON sessions(token);
CREATE INDEX idx_applications_user_id ON applications(user_id);
CREATE INDEX idx_deployments_app_id ON deployments(application_id);
```

### 7.2 Run Database Migrations

```bash
# Run migrations
psql -U platform_user -d platform_dev -f platform/config/schema.sql

# Verify tables created
psql -U platform_user -d platform_dev -c "\dt"
```

---

## Step 8: Monitoring Setup

### 8.1 Configure Health Checks

```bash
# Create health checks configuration
cat > platform/config/health-checks.json << 'EOF'
{
  "version": "1.0.0",
  "checks": [
    {
      "name": "api",
      "type": "http",
      "endpoint": "http://localhost:3000/health",
      "interval": 30000,
      "timeout": 5000,
      "threshold": 2
    },
    {
      "name": "database",
      "type": "tcp",
      "endpoint": "localhost:5432",
      "interval": 60000,
      "timeout": 3000
    },
    {
      "name": "redis",
      "type": "command",
      "endpoint": "redis-cli ping",
      "interval": 60000,
      "timeout": 3000
    }
  ]
}
EOF
```

### 8.2 Start Health Monitor

```bash
# Start health monitoring
node resources/health-monitor.js start &

# Check status
node resources/health-monitor.js status
```

**Expected Output:**
```json
{
  "running": true,
  "checks": {
    "api": {
      "name": "api",
      "checks": 10,
      "successRate": "100.00%",
      "avgResponseTime": "45.23ms",
      "currentStatus": "healthy"
    },
    "database": {
      "name": "database",
      "checks": 10,
      "successRate": "100.00%",
      "currentStatus": "healthy"
    }
  },
  "overall": "healthy"
}
```

---

## Step 9: Application Deployment

### 9.1 Create Sample Application

```javascript
// File: platform/services/app.js
const express = require('express');
const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    version: '1.0.0'
  });
});

// API info endpoint
app.get('/api/info', (req, res) => {
  res.json({
    name: 'Flow Nexus Platform API',
    version: '1.0.0',
    environment: process.env.NODE_ENV,
    services: {
      database: 'connected',
      redis: 'connected',
      storage: 'available'
    }
  });
});

// Start server
app.listen(port, () => {
  console.log(`Platform API running on port ${port}`);
  console.log(`Environment: ${process.env.NODE_ENV}`);
  console.log(`Health check: http://localhost:${port}/health`);
});

module.exports = app;
```

### 9.2 Deploy Application

```bash
# Deploy using deployment manager
python3 resources/deployment-manager.py deploy \
  platform-api \
  1.0.0 \
  platform/services \
  /var/www/platform

# Check deployment status
python3 resources/deployment-manager.py status platform-api
```

---

## Step 10: Verification and Testing

### 10.1 Run Test Suite

```bash
# Run all tests
bash tests/test-platform-init.sh
python3 tests/test-service-orchestrator.py
node tests/test-health-monitor.js
```

### 10.2 Verify Platform Health

```bash
# Check platform health
curl http://localhost:3000/health

# Get API info
curl http://localhost:3000/api/info

# Check service status
python3 resources/service-orchestrator.py status
```

### 10.3 Verify Monitoring

```bash
# Check health monitoring
node resources/health-monitor.js status

# View metrics
curl http://localhost:9090/metrics
```

---

## Next Steps

1. **Customize Configuration**: Adjust settings in `platform/config/flow-nexus.json`
2. **Add More Services**: Extend `platform/config/services.json`
3. **Configure Scaling**: Update deployment strategies
4. **Setup Alerts**: Configure monitoring alerts
5. **Deploy to Production**: Follow production deployment guide

---

## Troubleshooting

### Issue: Sandbox Creation Fails

```bash
# Check Flow Nexus connection
npx flow-nexus@latest --version

# Verify credentials
npx flow-nexus@latest profile

# Check logs
cat logs/platform/sandbox-errors.log
```

### Issue: Database Connection Fails

```bash
# Test database connection
psql -U platform_user -d platform_dev -c "SELECT 1"

# Check database status
pg_isready -h localhost -p 5432

# Review logs
cat logs/platform/database.log
```

### Issue: Health Checks Failing

```bash
# Check service status
python3 resources/service-orchestrator.py status

# Restart services
python3 resources/service-orchestrator.py restart api

# View detailed logs
node resources/health-monitor.js metrics
```

---

## Summary

You've successfully:
- ✅ Initialized Flow Nexus platform
- ✅ Configured all services
- ✅ Set up authentication
- ✅ Created sandboxes and storage
- ✅ Deployed database schema
- ✅ Configured monitoring
- ✅ Deployed sample application
- ✅ Verified platform health

**Platform Status**: Fully Operational
**Total Setup Time**: ~30 minutes
**Services Running**: API, Database, Redis, Monitoring
