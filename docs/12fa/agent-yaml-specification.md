# Agent Manifest Specification (agent.yaml)

**Version:** 1.0.0
**Status:** Production Ready
**Last Updated:** 2025-11-01

## Overview

The Agent Manifest (`agent.yaml`) is a comprehensive specification for defining 12-Factor compliant AI agents in the Claude Flow ecosystem. This specification ensures agents are portable, scalable, observable, and production-ready.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Metadata](#core-metadata)
3. [The 12 Factors](#the-12-factors)
4. [Extended Features](#extended-features)
5. [Validation](#validation)
6. [Migration Guide](#migration-guide)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

## Quick Start

### Minimal Agent Manifest

```yaml
name: my-agent
version: 1.0.0
purpose: "Brief description of what this agent does"
agent_type: coder

codebase:
  type: inline
  inline_prompt: "You are a coding agent that..."

config:
  schema:
    type: object
  env_vars_required: []
  env_vars_optional: []
```

### Full-Featured Agent

See `schemas/agent-manifest-v1.example.yaml` for a comprehensive example demonstrating all features.

## Core Metadata

### Required Fields

#### `name` (required)
- **Type:** `string`
- **Pattern:** `^[a-z][a-z0-9-]*$` (kebab-case)
- **Min Length:** 2
- **Max Length:** 64
- **Description:** Unique identifier for the agent
- **Examples:** `researcher`, `system-architect`, `code-reviewer`

#### `version` (required)
- **Type:** `string`
- **Pattern:** Semantic versioning (semver 2.0.0)
- **Description:** Agent version following semver specification
- **Examples:** `1.0.0`, `2.1.3-alpha`, `1.0.0-beta.1`

#### `purpose` (required)
- **Type:** `string`
- **Min Length:** 10
- **Max Length:** 500
- **Description:** Clear statement of the agent's responsibility
- **Best Practice:** Focus on the "why" and "what value" rather than "how"

#### `agent_type` (required)
- **Type:** `string`
- **Enum:** `researcher`, `coder`, `reviewer`, `tester`, `planner`, `system-architect`, `coordinator`, etc.
- **Description:** Primary classification for role-based coordination

### Optional Metadata

#### `metadata` (optional)
Additional contextual information:

```yaml
metadata:
  category: development          # development, testing, coordination, etc.
  complexity: moderate            # simple, moderate, complex, expert
  created: "2024-01-15"          # ISO date
  updated: "2025-10-29"          # ISO date
  author: "Your Team"
  license: MIT
  tags:
    - research
    - analysis
```

## The 12 Factors

### Factor 1: Codebase

**Principle:** One codebase tracked in revision control, many deploys

```yaml
codebase:
  type: git | npm | local | inline

  # For Git repositories
  repository: https://github.com/org/repo
  branch: main
  commit: <sha1-hash>
  path: agents/researcher

  # For NPM packages
  npm_package: "@scope/package@version"

  # For local development
  path: /path/to/agent

  # For inline agents
  inline_prompt: "Agent prompt content..."
```

**Types:**
- **git:** Agent code in Git repository (recommended for production)
- **npm:** Published NPM package
- **local:** Local filesystem path (development only)
- **inline:** Prompt defined directly in manifest (simple agents)

### Factor 2: Dependencies

**Principle:** Explicitly declare and isolate dependencies

```yaml
dependencies:
  npm_packages:
    - name: axios
      version: "^1.6.0"
      required: true
      scope: runtime | development | peer

  system_packages:
    - name: git
      version: ">=2.30.0"
      platform: [linux, darwin, win32, all]
      installer: apt | brew | choco | manual

  mcp_servers:
    - name: claude-flow
      command: npx claude-flow@alpha mcp start
      required: true
      capabilities:
        - memory_store
        - swarm_coordination

  other_agents:
    - agent: planner
      relationship: requires | collaborates | delegates | coordinates
      optional: false

  api_dependencies:
    - name: github-api
      endpoint: https://api.github.com
      version: v3
      authentication: none | api_key | oauth2 | bearer | basic
      required: false
```

**Dependency Types:**
1. **npm_packages:** JavaScript/Node.js dependencies
2. **system_packages:** OS-level binaries and tools
3. **mcp_servers:** MCP server dependencies for tools/coordination
4. **other_agents:** Agent-to-agent dependencies
5. **api_dependencies:** External API services

### Factor 3: Config

**Principle:** Store config in the environment

```yaml
config:
  # JSON Schema for configuration validation
  schema:
    type: object
    properties:
      max_retries:
        type: integer
        minimum: 1
        maximum: 10
        default: 3

  # Required environment variables
  env_vars_required:
    - name: ANTHROPIC_API_KEY
      description: "API key for Claude access"
      type: string | number | boolean | json | url | path
      secret: true
      validation:
        pattern: "^sk-ant-[a-zA-Z0-9-_]+$"

  # Optional environment variables with defaults
  env_vars_optional:
    - name: LOG_LEVEL
      default: "info"
      description: "Logging verbosity"
      type: string

  # Configuration files (minimize usage)
  config_files:
    - path: config/settings.json
      format: json | yaml | toml | env | ini
      optional: true
      schema: {...}

  strict_mode: true  # Fail on missing required config
```

**Best Practices:**
- ✅ Use environment variables for all configuration
- ✅ Mark sensitive values as `secret: true`
- ✅ Provide clear descriptions and validation rules
- ✅ Use `strict_mode: true` to catch config errors early
- ❌ Avoid config files when possible
- ❌ Never hardcode credentials or secrets

### Factor 4: Backing Services

**Principle:** Treat backing services as attached resources

```yaml
backing_services:
  databases:
    - name: agent-memory
      type: postgresql | mysql | mongodb | redis | sqlite | agentdb
      url_env_var: DATABASE_URL
      connection_pool:
        min: 1
        max: 10
      required: true

  caches:
    - name: research-cache
      type: redis | memcached | in-memory
      url_env_var: REDIS_URL
      ttl: 3600

  message_queues:
    - name: task-queue
      type: rabbitmq | kafka | redis-queue | sqs | pubsub
      url_env_var: QUEUE_URL
      topics:
        - agent.requests
        - agent.results

  storage:
    - name: artifacts
      type: s3 | gcs | azure-blob | local-fs
      bucket_env_var: STORAGE_BUCKET

  external_services:
    - name: github-api
      type: rest-api
      url_env_var: GITHUB_API_URL
      health_check: https://api.github.com/status
```

**Key Concepts:**
- All service connections via environment variables
- Services are interchangeable (dev/staging/prod)
- Health checks for service availability
- Connection pooling for databases

### Factor 5: Build, Release, Run

**Principle:** Strictly separate build and run stages

```yaml
build_release_run:
  build:
    commands:
      - npm install
      - npm run build
      - npm test
    artifacts:
      - dist/
      - node_modules/
    cache_key: package-lock.json

  release:
    versioning: semver | git-sha | timestamp | build-number
    immutable: true
    packaging: npm | docker | binary | none

  run:
    entrypoint: node dist/index.js
    startup_timeout: 30
    health_check:
      command: curl -f http://localhost:3000/health
      interval: 30
      retries: 3
```

**Stages:**
1. **Build:** Transform code into executables (compile, bundle, test)
2. **Release:** Combine build with config for specific deployment
3. **Run:** Execute the release in the target environment

### Factor 6: Processes

**Principle:** Execute as one or more stateless processes

```yaml
processes:
  stateless: true
  process_model: single | worker-pool | event-driven | request-response

  session_management:
    type: none | external-store | sticky-sessions
    backend: redis | memcached | database

  memory_model:
    shared_nothing: true
    max_memory: 512Mi
```

**Process Models:**
- **single:** One process handles all work
- **worker-pool:** Multiple workers process queue
- **event-driven:** Event loop with async handlers
- **request-response:** Synchronous request handling

**Statelessness:**
- No local state between requests
- Session data in external stores
- Horizontal scaling without coordination

### Factor 7: Port Binding

**Principle:** Export services via port binding

```yaml
port_binding:
  http:
    enabled: true
    port_env_var: PORT
    default_port: 3000
    routes:
      - path: /health
        method: GET
        description: Health check
      - path: /task
        method: POST
        description: Submit task

  grpc:
    enabled: false
    port_env_var: GRPC_PORT
    services: []

  websocket:
    enabled: true
    port_env_var: WS_PORT

  mcp:
    enabled: true
    transport: stdio | http | websocket
    tools:
      - name: agent_execute
        description: Execute agent task
        parameters: {...}
```

**Protocols Supported:**
- **HTTP/REST:** Standard web API
- **gRPC:** High-performance RPC
- **WebSocket:** Real-time bidirectional
- **MCP:** Model Context Protocol for tools

### Factor 8: Concurrency

**Principle:** Scale out via the process model

```yaml
concurrency:
  horizontal_scaling:
    enabled: true
    min_instances: 1
    max_instances: 10
    scaling_metric: cpu | memory | queue-depth | request-rate | custom
    target_value: 5

  concurrency_model: async-await | worker-threads | cluster | event-loop
  max_concurrent_tasks: 5
  load_balancing: round-robin | least-connections | random | weighted
```

**Scaling Strategies:**
- **Horizontal:** Add more process instances
- **Vertical:** Increase resources per instance (not recommended)
- **Auto-scaling:** Based on metrics and thresholds

### Factor 9: Disposability

**Principle:** Fast startup and graceful shutdown

```yaml
disposability:
  startup:
    target_time: 5          # seconds
    initialization:
      - Load configuration
      - Connect to services
      - Initialize caches
    lazy_loading: true

  shutdown:
    graceful_timeout: 30    # seconds
    cleanup_steps:
      - Finish in-progress tasks
      - Flush caches
      - Close connections
    signals:
      - SIGTERM
      - SIGINT

  crash_recovery:
    auto_restart: true
    max_restarts: 3
    backoff_strategy: exponential | linear | constant
```

**Best Practices:**
- ✅ Start quickly (< 10 seconds)
- ✅ Handle SIGTERM for graceful shutdown
- ✅ Complete in-flight work before terminating
- ✅ Use exponential backoff for restarts
- ❌ Don't rely on long startup times
- ❌ Don't assume process will run forever

### Factor 10: Dev/Prod Parity

**Principle:** Keep environments as similar as possible

```yaml
dev_prod_parity:
  environment_matrix:
    development:
      backing_services:
        databases:
          agent-memory: sqlite://./dev.db
      config_overrides:
        LOG_LEVEL: debug

    staging:
      backing_services:
        databases:
          agent-memory: postgresql://staging-db/agents
      config_overrides:
        LOG_LEVEL: info

    production:
      backing_services:
        databases:
          agent-memory: postgresql://prod-db/agents
      config_overrides:
        LOG_LEVEL: warn

  parity_enforcement:
    same_dependencies: true
    same_backing_services: true
    deployment_frequency: continuous | daily | weekly
```

**Gaps to Minimize:**
- **Time gap:** Deploy frequently (hours, not months)
- **Personnel gap:** Developers deploy their own code
- **Tools gap:** Same services in dev and prod

### Factor 11: Logs

**Principle:** Treat logs as event streams

```yaml
logs:
  output: stdout | stderr | file | syslog
  format: json | text | structured
  level: debug | info | warn | error | fatal

  structured_fields:
    - timestamp
    - level
    - agent
    - task_id
    - correlation_id

  routing:
    aggregator: none | elasticsearch | splunk | datadog | cloudwatch
    sampling_rate: 1.0

  sensitive_data:
    redaction: true
    patterns:
      - "(?i)(api[_-]?key|token)[\"']?\\s*[:=]\\s*[\"']?([^\\s\"']+)"
```

**Best Practices:**
- ✅ Write to stdout/stderr (not files)
- ✅ Use structured JSON format
- ✅ Include correlation IDs for tracing
- ✅ Redact sensitive data automatically
- ❌ Don't manage log rotation
- ❌ Don't write to local files

### Factor 12: Admin Processes

**Principle:** Run admin tasks as one-off processes

```yaml
admin_processes:
  tasks:
    - name: migrate-database
      command: node scripts/migrate.js
      description: Run database migrations
      requires_approval: true
      idempotent: false

    - name: clear-cache
      command: node scripts/clear-cache.js
      description: Clear all caches
      schedule: "0 2 * * *"  # Cron expression
      idempotent: true

  console_access:
    enabled: true
    command: node --experimental-repl-await scripts/console.js

  migrations:
    enabled: true
    directory: migrations/
    auto_run: false
```

**Task Types:**
- **One-off:** Manual execution (migrations, repairs)
- **Scheduled:** Cron-based recurring tasks
- **Console:** Interactive REPL access

## Extended Features

### Observability

```yaml
observability:
  metrics:
    enabled: true
    endpoint: /metrics
    format: prometheus | statsd | json
    custom_metrics:
      - name: tasks_completed_total
        type: counter | gauge | histogram | summary
        description: Total tasks completed

  tracing:
    enabled: true
    provider: jaeger | zipkin | opentelemetry
    sampling_rate: 0.1

  health_checks:
    liveness:
      endpoint: /health/live
      checks:
        - process-running
    readiness:
      endpoint: /health/ready
      checks:
        - database-connected
        - cache-available
```

### Capabilities

```yaml
capabilities:
  primary_skills:
    - Code analysis
    - Architecture design
  secondary_skills:
    - Documentation
    - Testing
  tools:
    - bash
    - git
    - npm
  languages:
    - JavaScript
    - TypeScript
    - Python
  frameworks:
    - Node.js
    - React
```

### Coordination

```yaml
coordination:
  topology: mesh | hierarchical | ring | star | pipeline

  communication:
    protocols:
      - http
      - message-queue
      - memory
    message_format: json | protobuf | messagepack

  hooks:
    pre_task: true
    post_task: true
    post_edit: true
    notify: true
```

### Constraints

```yaml
constraints:
  resource_limits:
    max_memory: 512Mi
    max_cpu: "1.0"
    max_execution_time: 300
    max_file_size: 10Mi

  rate_limits:
    requests_per_minute: 60
    concurrent_operations: 5

  security:
    sandbox: true
    network_access: full | restricted | none
    file_system_access: full | workspace-only | read-only | none
```

### Testing

```yaml
testing:
  test_commands:
    - npm test
    - npm run test:integration
  coverage_threshold: 80
  test_patterns:
    - "**/*.test.js"
    - "**/*.spec.js"
```

## Validation

### Schema Validation

The manifest is validated against `schemas/agent-manifest-v1.json` using JSON Schema Draft 7.

```bash
# Validate a manifest
npx ajv validate \
  -s schemas/agent-manifest-v1.json \
  -d my-agent.yaml
```

### Runtime Validation

Agents validate their configuration at startup:

```javascript
const Ajv = require('ajv');
const yaml = require('js-yaml');
const fs = require('fs');

const schema = require('./schemas/agent-manifest-v1.json');
const manifest = yaml.load(fs.readFileSync('./agent.yaml', 'utf8'));

const ajv = new Ajv();
const validate = ajv.compile(schema);
const valid = validate(manifest);

if (!valid) {
  console.error('Invalid manifest:', validate.errors);
  process.exit(1);
}
```

## Migration Guide

### From Existing Agents

#### Step 1: Create Manifest File

Create `agent.yaml` in your agent directory:

```yaml
name: my-existing-agent
version: 1.0.0
purpose: "What your agent does"
agent_type: <select-appropriate-type>
```

#### Step 2: Document Codebase

```yaml
codebase:
  type: git
  repository: <your-repo-url>
  path: agents/my-agent
```

#### Step 3: Extract Dependencies

Review your `package.json` and system requirements:

```yaml
dependencies:
  npm_packages:
    # Copy from package.json dependencies
  mcp_servers:
    # List MCP servers you use
```

#### Step 4: Document Configuration

Identify all environment variables and config files:

```yaml
config:
  env_vars_required:
    - name: REQUIRED_VAR
      description: "Purpose"
      type: string
  env_vars_optional:
    - name: OPTIONAL_VAR
      default: "value"
```

#### Step 5: Add Backing Services

List databases, caches, queues, etc.:

```yaml
backing_services:
  databases:
    - name: primary
      type: postgresql
      url_env_var: DATABASE_URL
```

#### Step 6: Complete Remaining Factors

Work through factors 5-12, adding configuration as appropriate.

### Backward Compatibility

Existing agents without manifests continue to work. The manifest is additive:

- ✅ Old agents work without changes
- ✅ New agents use manifest for enhanced features
- ✅ Mixed deployments supported
- ✅ Gradual migration path

## Best Practices

### 1. Start Small, Grow Incrementally

```yaml
# Minimal viable manifest
name: my-agent
version: 1.0.0
purpose: "Agent purpose"
agent_type: coder
codebase:
  type: inline
  inline_prompt: "You are..."
config:
  schema: {type: object}
  env_vars_required: []
```

Add factors as needed for your use case.

### 2. Version Your Manifests

```yaml
# Include manifest version in agent version
version: 1.2.0  # Agent v1.2.0 with manifest v1
```

### 3. Document Everything

```yaml
# Use descriptions liberally
purpose: "Clear, detailed explanation of value provided"

env_vars_required:
  - name: API_KEY
    description: "Obtained from https://platform.example.com/api-keys"
```

### 4. Validate Early and Often

```bash
# Add to CI/CD
npm run validate-manifest
npm run test-config
```

### 5. Use Semantic Versioning

```yaml
version: 1.2.3
# 1 = major (breaking changes)
# 2 = minor (new features)
# 3 = patch (bug fixes)
```

### 6. Minimize Config Files

```yaml
# Prefer this
config:
  env_vars_required:
    - name: MAX_RETRIES
      type: number

# Over this
config:
  config_files:
    - path: config.json  # Avoid when possible
```

### 7. Design for Disposability

```yaml
disposability:
  startup:
    target_time: 5      # Fast startup
    lazy_loading: true  # Load on demand
  shutdown:
    graceful_timeout: 30  # Clean shutdown
```

### 8. Implement Health Checks

```yaml
observability:
  health_checks:
    liveness:
      endpoint: /health/live
      checks: [process-running]
    readiness:
      endpoint: /health/ready
      checks: [database-connected, cache-available]
```

### 9. Enable Observability

```yaml
observability:
  metrics:
    enabled: true
  tracing:
    enabled: true
  logs:
    format: json
    structured_fields: [timestamp, level, agent, task_id]
```

### 10. Test Across Environments

```yaml
dev_prod_parity:
  environment_matrix:
    development: {...}
    staging: {...}
    production: {...}
  parity_enforcement:
    same_dependencies: true
```

## Examples

### Example 1: Simple Inline Agent

```yaml
name: hello-world
version: 1.0.0
purpose: "Greets users and provides helpful responses"
agent_type: coder

codebase:
  type: inline
  inline_prompt: "You are a friendly assistant that greets users warmly."

config:
  schema:
    type: object
  env_vars_required: []
```

### Example 2: Production Researcher Agent

See `schemas/agent-manifest-v1.example.yaml` for a comprehensive production example.

### Example 3: Multi-Service Backend Agent

```yaml
name: api-orchestrator
version: 2.0.0
purpose: "Orchestrates multiple backend services for complex operations"
agent_type: backend-dev

codebase:
  type: npm
  npm_package: "@company/api-orchestrator@2.0.0"

dependencies:
  npm_packages:
    - {name: express, version: "^4.18.0"}
    - {name: pg, version: "^8.11.0"}
  mcp_servers:
    - {name: claude-flow, command: "npx claude-flow@alpha mcp start"}

config:
  env_vars_required:
    - {name: DATABASE_URL, type: url}
    - {name: REDIS_URL, type: url}
    - {name: API_KEY, type: string, secret: true}

backing_services:
  databases:
    - {name: primary, type: postgresql, url_env_var: DATABASE_URL}
  caches:
    - {name: session, type: redis, url_env_var: REDIS_URL}
  message_queues:
    - {name: tasks, type: redis-queue, url_env_var: REDIS_URL}

port_binding:
  http:
    enabled: true
    port_env_var: PORT
    default_port: 3000

concurrency:
  horizontal_scaling:
    enabled: true
    min_instances: 2
    max_instances: 20
    scaling_metric: request-rate
    target_value: 100

observability:
  metrics:
    enabled: true
  tracing:
    enabled: true
  health_checks:
    readiness:
      checks: [database-connected, redis-available]
```

## Tooling

### Validation Tools

```bash
# JSON Schema validation
npm install -g ajv-cli
ajv validate -s schemas/agent-manifest-v1.json -d agent.yaml

# Claude Flow validation
npx claude-flow@alpha agent validate agent.yaml

# Runtime validation in your agent
const {validateManifest} = require('@claude-flow/agent-sdk');
validateManifest(manifest);
```

### Generation Tools

```bash
# Generate manifest from existing agent
npx claude-flow@alpha agent generate-manifest

# Interactive manifest creator
npx claude-flow@alpha agent init
```

### Documentation Generation

```bash
# Generate docs from manifest
npx claude-flow@alpha agent docs agent.yaml

# Generate OpenAPI spec from manifest
npx claude-flow@alpha agent openapi agent.yaml
```

## Additional Resources

- [12-Factor App Methodology](https://12factor.net/)
- [Semantic Versioning](https://semver.org/)
- [JSON Schema Documentation](https://json-schema.org/)
- [Claude Flow Documentation](https://github.com/ruvnet/claude-flow)
- [Agent Development Guide](./agent-development-guide.md)

## FAQ

### Q: Do I need to implement all 12 factors?

**A:** No. Start with the core factors (1-3) and add others as your agent's requirements grow. Simple agents may only need a few factors.

### Q: Can I use this with non-JavaScript agents?

**A:** Yes! The specification is language-agnostic. Adapt the `dependencies` and `build_release_run` sections for your language.

### Q: How do I handle secrets?

**A:** Always use environment variables with `secret: true`. Never commit secrets to version control. Use secret management services in production.

### Q: What if my agent needs to maintain state?

**A:** Store state in external backing services (Factor 4) like databases or Redis. The agent process itself should remain stateless (Factor 6).

### Q: Can agents share a manifest?

**A:** No. Each agent should have its own manifest defining its specific requirements and configuration.

### Q: How do I test manifest changes?

**A:** Validate with JSON Schema, test in development environment, then promote through staging to production following Factor 10 (dev/prod parity).

## Version History

- **1.0.0** (2025-11-01): Initial production release
  - Complete 12-factor specification
  - Extended features (observability, capabilities, coordination)
  - Comprehensive validation rules
  - Migration guide and examples

---

**Specification Status:** ✅ Production Ready
**Schema Location:** `schemas/agent-manifest-v1.json`
**Example Location:** `schemas/agent-manifest-v1.example.yaml`
