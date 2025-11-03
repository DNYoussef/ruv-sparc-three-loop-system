# Flow Nexus Platform Resources

Supporting scripts, templates, and utilities for the Flow Nexus Platform skill.

## Directory Structure

```
resources/
├── scripts/           # Automation scripts for platform operations
│   ├── auth-manager.js          # Authentication & user management
│   ├── sandbox-manager.js       # Sandbox lifecycle management
│   ├── deployment-manager.js    # Application deployment automation
│   └── platform-health.js       # System health monitoring
│
└── templates/        # Configuration templates
    ├── platform-config.json     # Complete platform configuration
    ├── sandbox-config.yaml      # Sandbox setup template
    └── deployment-manifest.yaml # Deployment configuration
```

## Scripts

### auth-manager.js

**Purpose**: Automate authentication workflows via Flow Nexus MCP tools.

**Features**:
- User registration and verification
- Login/logout with session management
- Password reset and updates
- Profile management
- Tier upgrades (free → pro → enterprise)

**Usage**:
```bash
node auth-manager.js register <email> <password> [fullName]
node auth-manager.js login <email> <password>
node auth-manager.js status [--detailed]
node auth-manager.js update-profile <userId> <key=value> [key=value...]
```

**Requirements**: Node.js 14+, Flow Nexus MCP server running

---

### sandbox-manager.js

**Purpose**: Manage complete sandbox lifecycle.

**Features**:
- Multi-template sandbox creation (node, python, react, etc.)
- Environment variable and package configuration
- Code execution (inline or from files)
- File uploads to sandboxes
- Log retrieval
- Automated cleanup

**Usage**:
```bash
node sandbox-manager.js create <template> <name> [--env KEY=VALUE] [--packages pkg1,pkg2]
node sandbox-manager.js execute <sandboxId> <code|@file>
node sandbox-manager.js cleanup-all [--older-than-hours N]
```

**Requirements**: Node.js 14+, Flow Nexus MCP server running

---

### deployment-manager.js

**Purpose**: Automate application deployment and app store publishing.

**Features**:
- Template browsing and search
- Application deployment with variable substitution
- App publishing to store
- Analytics tracking
- Template management

**Usage**:
```bash
node deployment-manager.js list-templates [--category <cat>] [--featured]
node deployment-manager.js deploy <templateName> <deploymentName> [--var KEY=VALUE]
node deployment-manager.js publish <name> <description> <category> <sourceFile> [--tags tag1,tag2]
```

**Requirements**: Node.js 14+, Flow Nexus MCP server running

---

### platform-health.js

**Purpose**: Monitor system health and manage credits.

**Features**:
- System health checks with component status
- Credit balance tracking
- Payment link generation
- Auto-refill configuration
- Audit log access
- User statistics
- Market data analytics

**Usage**:
```bash
node platform-health.js check [--detailed]
node platform-health.js credits [--history]
node platform-health.js payment-link <amount>
node platform-health.js auto-refill <enable|disable> [--threshold N] [--amount N]
```

**Requirements**: Node.js 14+, Flow Nexus MCP server running

---

## Templates

### platform-config.json

Complete platform configuration including:
- Authentication settings (session timeout, MFA, password policy)
- Sandbox defaults (templates, timeouts, resource limits)
- Deployment configuration (region, auto-deploy, rollback)
- Credit management (auto-refill, spending limits)
- Monitoring setup (health checks, metrics, alerts)
- Integrations (GitHub, Slack, webhooks)

**Usage**: Customize for your platform instance and apply via MCP tools.

---

### sandbox-config.yaml

Comprehensive sandbox configuration template:
- Environment variables and secrets
- Package installation
- Startup scripts and lifecycle hooks
- Network configuration
- Storage and volume mounts
- Security settings
- Monitoring and health checks

**Usage**: Use as base for custom sandbox templates.

---

### deployment-manifest.yaml

Enterprise deployment configuration:
- Template selection and versioning
- Resource requests and limits
- Autoscaling configuration
- Health probes (liveness, readiness)
- Database and cache setup
- Monitoring and observability
- CI/CD integration
- Security policies

**Usage**: Define complex deployments with full infrastructure as code.

---

## Common Workflows

### Complete Onboarding

```bash
# 1. Register and login
node auth-manager.js register dev@example.com SecurePass123! "Developer Name"
node auth-manager.js login dev@example.com SecurePass123!

# 2. Check credits and setup auto-refill
node platform-health.js credits
node platform-health.js auto-refill enable --threshold 100 --amount 50

# 3. Create development sandbox
node sandbox-manager.js create node dev-sandbox --env NODE_ENV=development --packages express,cors,dotenv

# 4. Deploy application
node deployment-manager.js deploy express-api-starter my-production-api --var database_url=postgres://...
```

### Sandbox Development Cycle

```bash
# 1. Create sandbox
node sandbox-manager.js create node api-dev --packages express,jest

# 2. Upload files
node sandbox-manager.js upload sbx_123 ./package.json /app/package.json
node sandbox-manager.js upload sbx_123 ./src/server.js /app/src/server.js

# 3. Execute tests
node sandbox-manager.js execute sbx_123 @run-tests.sh

# 4. View logs
node sandbox-manager.js logs sbx_123 --lines 50

# 5. Cleanup
node sandbox-manager.js stop sbx_123
node sandbox-manager.js delete sbx_123
```

### Deployment Pipeline

```bash
# 1. Browse templates
node deployment-manager.js list-templates --category web-api

# 2. Get template info
node deployment-manager.js template-info express-api-starter

# 3. Deploy to staging
node deployment-manager.js deploy express-api-starter my-app-staging --var database_url=postgres://staging

# 4. Verify deployment
node platform-health.js check --detailed

# 5. Deploy to production
node deployment-manager.js deploy express-api-starter my-app-production --var database_url=postgres://production
```

---

## Error Handling

All scripts include comprehensive error handling:
- Parameter validation
- MCP call failures with retry logic
- Graceful degradation
- Detailed error messages
- Non-zero exit codes on failure

---

## Integration

These scripts work seamlessly with:
- Flow Nexus MCP server (`mcp__flow-nexus__*` tools)
- Claude Code workflows
- CI/CD pipelines
- Automated deployment systems
- Monitoring and alerting platforms

---

## Support

For issues or questions:
- Documentation: `../SKILL.md`
- Flow Nexus Docs: https://docs.flow-nexus.ruv.io
- MCP Server: Ensure `mcp__flow-nexus__*` tools are available
- GitHub: https://github.com/ruvnet/flow-nexus

---

**Last Updated**: 2025-11-02
**Version**: 2.0.0 (Gold Tier)
