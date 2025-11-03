# Example 2: Task Orchestration - Distributed Microservices Development

## Scenario Overview

Continuing from Example 1, you now have a functioning swarm with 6 specialized agents. Your goal is to orchestrate the development of the e-commerce platform's core microservices by delegating tasks to the appropriate agents based on their capabilities and coordinating their work to ensure integration.

**Microservices to Build:**
1. **Authentication Service** - OAuth 2.0 + JWT (Backend Auth Specialist)
2. **Product Catalog Service** - GraphQL API (Backend Catalog Specialist)
3. **Database Schema** - PostgreSQL design (Database Architect)
4. **API Gateway** - OpenAPI spec (API Designer)
5. **Security Audit** - OWASP compliance (Security Specialist)
6. **Container Deployment** - Docker + K8s (DevOps Engineer)

**Success Criteria:**
- All services deployed to cloud sandboxes
- API contracts validated between services
- Security audit passes with 0 critical vulnerabilities
- 90%+ test coverage across services
- CI/CD pipeline operational

---

## Step-by-Step Walkthrough

### Phase 1: High-Priority Authentication Service (Critical Path)

**Why Start Here?** Auth service is a dependency for all other services. Use **critical priority** to ensure fastest execution.

```javascript
// Step 1: Orchestrate authentication service development
mcp__flow-nexus__task_orchestrate({
  task: `Build OAuth 2.0 + JWT authentication service with the following requirements:

  1. Implement OAuth 2.0 authorization code flow with PKCE
  2. JWT token generation with RS256 signing (public/private key pair)
  3. Refresh token rotation with secure storage
  4. Rate limiting: 10 login attempts per minute per IP
  5. Password hashing with bcrypt (cost factor 12)
  6. Session management with Redis
  7. Email verification workflow
  8. Password reset with time-limited tokens
  9. Unit tests with 95%+ coverage
  10. Integration tests for token lifecycle

  Deliverables:
  - /src/auth-service/server.js (Express.js server)
  - /src/auth-service/routes/auth.js (Auth endpoints)
  - /src/auth-service/middleware/jwt-validator.js
  - /src/auth-service/config/oauth-config.js
  - /tests/auth-service/auth.test.js
  - /docs/auth-service/API.md (OpenAPI spec)`,

  strategy: "adaptive",        // Route to best-performing agent
  priority: "critical",        // Highest priority execution
  maxAgents: 2                 // Backend Auth Specialist + Tester
})
```

**Expected Output:**
```json
{
  "task_id": "task-auth-service-001",
  "status": "queued",
  "strategy": "adaptive",
  "priority": "critical",
  "assigned_agents": [
    "agent-backend-auth-specialist-a1b2c3"
  ],
  "estimated_completion_time_minutes": 45,
  "created_at": "2025-11-02T11:00:00Z"
}
```

**What Happened:**
- ✅ Task queued with critical priority (jumps to front of queue)
- ✅ Adaptive strategy selected best agent: Backend Auth Specialist
- ✅ Estimated 45 minutes for implementation + testing
- ✅ Task ID generated for status tracking

---

### Phase 2: Parallel Product Catalog Service (Independent Task)

**Why Parallel?** Catalog service doesn't depend on auth initially. Use **parallel strategy** for concurrent development.

```javascript
// Step 2: Orchestrate product catalog service (parallel to auth)
mcp__flow-nexus__task_orchestrate({
  task: `Build GraphQL product catalog service with REST fallback:

  GraphQL Schema:
  - Product type (id, name, description, price, inventory, images)
  - Query: products(filter, sort, pagination)
  - Query: product(id)
  - Mutation: createProduct (admin only)
  - Mutation: updateProduct (admin only)
  - Subscription: inventoryUpdated

  REST Endpoints:
  - GET /api/v1/products (with filtering, sorting, pagination)
  - GET /api/v1/products/:id
  - POST /api/v1/products (admin, protected)
  - PUT /api/v1/products/:id (admin, protected)

  Requirements:
  - Redis caching for product queries (5-minute TTL)
  - ElasticSearch integration for full-text search
  - Image upload to S3 with CDN URLs
  - GraphQL DataLoader for batch queries
  - Rate limiting: 100 requests/minute per user

  Deliverables:
  - /src/catalog-service/server.js (Express + Apollo Server)
  - /src/catalog-service/graphql/schema.graphql
  - /src/catalog-service/graphql/resolvers.js
  - /src/catalog-service/rest/routes.js
  - /src/catalog-service/cache/redis-client.js
  - /tests/catalog-service/graphql.test.js`,

  strategy: "parallel",        // Execute concurrently with auth task
  priority: "high",            // High priority but not critical
  maxAgents: 2                 // Backend Catalog Specialist + Database Architect
})
```

**Expected Output:**
```json
{
  "task_id": "task-catalog-service-002",
  "status": "running",
  "strategy": "parallel",
  "priority": "high",
  "assigned_agents": [
    "agent-backend-catalog-specialist-d4e5f6",
    "agent-database-architect-g7h8i9"
  ],
  "estimated_completion_time_minutes": 60,
  "created_at": "2025-11-02T11:01:00Z"
}
```

**Parallel Execution:**
- ✅ Task running concurrently with auth service
- ✅ Two agents assigned: Catalog specialist + Database architect
- ✅ Database architect designs Product table schema in parallel
- ✅ Estimated 60 minutes (more complex than auth)

---

### Phase 3: Database Schema Design (Foundation Task)

**Why Sequential?** Schema must be finalized before services can connect. Use **sequential strategy** to ensure completion before migrations.

```javascript
// Step 3: Orchestrate database schema design (sequential)
mcp__flow-nexus__task_orchestrate({
  task: `Design comprehensive PostgreSQL database schema for e-commerce platform:

  Tables to Create:
  1. users (id, email, password_hash, email_verified, created_at, updated_at)
  2. sessions (id, user_id, token_hash, expires_at, ip_address, user_agent)
  3. refresh_tokens (id, user_id, token_hash, expires_at, revoked)
  4. products (id, name, description, price, inventory, images_json, created_at)
  5. categories (id, name, parent_id, slug)
  6. product_categories (product_id, category_id)
  7. orders (id, user_id, total, status, created_at, updated_at)
  8. order_items (id, order_id, product_id, quantity, price_snapshot)
  9. payments (id, order_id, amount, status, provider, transaction_id)

  Requirements:
  - Primary keys: BIGSERIAL for all tables
  - Foreign keys with ON DELETE CASCADE/RESTRICT as appropriate
  - Indexes: email (unique), product name (GIN for full-text), orders.user_id
  - Partitioning: orders table by created_at (monthly partitions)
  - Constraints: CHECK price >= 0, CHECK inventory >= 0
  - JSONB columns for images and metadata
  - Timestamps: created_at, updated_at with triggers

  Deliverables:
  - /database/migrations/001_create_users.sql
  - /database/migrations/002_create_products.sql
  - /database/migrations/003_create_orders.sql
  - /database/schema-diagram.png (entity relationship)
  - /database/indexes.sql (index definitions)
  - /database/seed-data.sql (sample data for testing)`,

  strategy: "sequential",      // Complete before other tasks can use DB
  priority: "critical",        // Blocking dependency for all services
  maxAgents: 1                 // Database Architect only
})
```

**Expected Output:**
```json
{
  "task_id": "task-database-schema-003",
  "status": "queued",
  "strategy": "sequential",
  "priority": "critical",
  "assigned_agents": [
    "agent-database-architect-g7h8i9"
  ],
  "dependencies": [],
  "estimated_completion_time_minutes": 30,
  "created_at": "2025-11-02T11:02:00Z"
}
```

---

### Phase 4: API Gateway Design (Integration Task)

**Why After Services?** Gateway needs service endpoints to proxy. Use **adaptive strategy** with **medium priority** since it's not blocking development.

```javascript
// Step 4: Orchestrate API gateway design (after services are defined)
mcp__flow-nexus__task_orchestrate({
  task: `Design API Gateway with OpenAPI 3.1 specification:

  Gateway Routes:
  - /auth/* → Auth Service (localhost:3001)
  - /catalog/* → Catalog Service (localhost:3002)
  - /orders/* → Orders Service (localhost:3003)
  - /payments/* → Payments Service (localhost:3004)

  Features:
  - JWT validation middleware (shared across all routes)
  - Rate limiting per route (different limits per service)
  - Request logging with correlation IDs
  - Circuit breaker for service failures (3 failures = open circuit)
  - Response caching for GET requests (configurable TTL)
  - CORS configuration for frontend origins
  - API versioning support (/v1/, /v2/)

  OpenAPI Spec:
  - Complete path definitions for all routes
  - Request/response schemas with examples
  - Security schemes (Bearer JWT)
  - Error response schemas (4xx, 5xx)
  - Rate limit headers documentation

  Deliverables:
  - /api-gateway/openapi.yaml (OpenAPI 3.1 spec)
  - /api-gateway/server.js (Express gateway server)
  - /api-gateway/middleware/rate-limiter.js
  - /api-gateway/middleware/circuit-breaker.js
  - /api-gateway/config/routes.js
  - /docs/api-gateway/ARCHITECTURE.md`,

  strategy: "adaptive",        // Route to best designer
  priority: "medium",          // Can be done after services
  maxAgents: 2                 // API Designer + Backend Specialist
})
```

**Expected Output:**
```json
{
  "task_id": "task-api-gateway-004",
  "status": "pending",
  "strategy": "adaptive",
  "priority": "medium",
  "assigned_agents": [
    "agent-api-designer-j1k2l3"
  ],
  "dependencies": ["task-auth-service-001", "task-catalog-service-002"],
  "estimated_completion_time_minutes": 40,
  "created_at": "2025-11-02T11:03:00Z"
}
```

**Dependency Tracking:**
- ⏳ Waiting for auth and catalog services to define endpoints
- ✅ API designer can draft routes but needs actual service contracts
- ✅ Will auto-start when dependencies complete

---

### Phase 5: Security Audit (Quality Assurance Task)

**Why Last?** Audit requires completed code. Use **sequential strategy** after implementation.

```javascript
// Step 5: Orchestrate security audit (after implementation complete)
mcp__flow-nexus__task_orchestrate({
  task: `Perform comprehensive security audit on e-commerce platform:

  OWASP Top 10 Compliance Checks:
  1. Injection: SQL injection, NoSQL injection, command injection
  2. Broken Authentication: JWT validation, session management, password policies
  3. Sensitive Data Exposure: Encryption at rest, TLS, secret management
  4. XML External Entities: N/A (no XML processing)
  5. Broken Access Control: Authorization checks, role-based access
  6. Security Misconfiguration: Default credentials, error handling, headers
  7. XSS: Input sanitization, CSP headers, output encoding
  8. Insecure Deserialization: JSON parsing, object validation
  9. Using Components with Known Vulnerabilities: npm audit, dependency scanning
  10. Insufficient Logging & Monitoring: Audit trails, security events

  Automated Scanning:
  - npm audit (dependency vulnerabilities)
  - Snyk scan (container vulnerabilities)
  - OWASP ZAP (dynamic application security testing)
  - SonarQube (static code analysis)
  - Trivy (Docker image scanning)

  Manual Code Review:
  - Authentication logic (JWT validation, token storage)
  - Authorization checks (role-based, resource-based)
  - Input validation (sanitization, type checking)
  - Cryptography usage (bcrypt, JWT signing, TLS)

  Deliverables:
  - /security/audit-report.md (executive summary)
  - /security/vulnerability-findings.json (structured data)
  - /security/remediation-plan.md (prioritized fixes)
  - /security/compliance-checklist.md (OWASP Top 10)
  - /security/scan-results/ (automated tool outputs)`,

  strategy: "sequential",      // Must run after all code is complete
  priority: "high",            // Critical for production readiness
  maxAgents: 1                 // Security Specialist only
})
```

**Expected Output:**
```json
{
  "task_id": "task-security-audit-005",
  "status": "pending",
  "strategy": "sequential",
  "priority": "high",
  "assigned_agents": [
    "agent-security-specialist-m4n5o6"
  ],
  "dependencies": [
    "task-auth-service-001",
    "task-catalog-service-002",
    "task-api-gateway-004"
  ],
  "estimated_completion_time_minutes": 90,
  "created_at": "2025-11-02T11:04:00Z"
}
```

---

### Phase 6: Check Task Progress (Real-Time Monitoring)

```javascript
// Step 6: Monitor all task statuses
mcp__flow-nexus__task_status({
  detailed: true
})
```

**Expected Output:**
```json
{
  "tasks": [
    {
      "task_id": "task-auth-service-001",
      "status": "running",
      "progress_percentage": 65,
      "assigned_agents": ["agent-backend-auth-specialist-a1b2c3"],
      "started_at": "2025-11-02T11:00:15Z",
      "estimated_completion": "2025-11-02T11:45:00Z",
      "current_step": "Writing integration tests"
    },
    {
      "task_id": "task-catalog-service-002",
      "status": "running",
      "progress_percentage": 40,
      "assigned_agents": [
        "agent-backend-catalog-specialist-d4e5f6",
        "agent-database-architect-g7h8i9"
      ],
      "started_at": "2025-11-02T11:01:30Z",
      "estimated_completion": "2025-11-02T12:01:00Z",
      "current_step": "Implementing GraphQL resolvers"
    },
    {
      "task_id": "task-database-schema-003",
      "status": "completed",
      "progress_percentage": 100,
      "assigned_agents": ["agent-database-architect-g7h8i9"],
      "started_at": "2025-11-02T11:02:00Z",
      "completed_at": "2025-11-02T11:32:00Z",
      "duration_minutes": 30
    },
    {
      "task_id": "task-api-gateway-004",
      "status": "pending",
      "progress_percentage": 0,
      "assigned_agents": ["agent-api-designer-j1k2l3"],
      "waiting_for_dependencies": [
        "task-auth-service-001",
        "task-catalog-service-002"
      ]
    },
    {
      "task_id": "task-security-audit-005",
      "status": "pending",
      "progress_percentage": 0,
      "assigned_agents": ["agent-security-specialist-m4n5o6"],
      "waiting_for_dependencies": [
        "task-auth-service-001",
        "task-catalog-service-002",
        "task-api-gateway-004"
      ]
    }
  ],
  "summary": {
    "total_tasks": 5,
    "completed": 1,
    "running": 2,
    "pending": 2,
    "failed": 0
  }
}
```

**Progress Insights:**
- ✅ Database schema completed in 30 minutes (on time!)
- 🏃 Auth service 65% done (integration tests in progress)
- 🏃 Catalog service 40% done (GraphQL resolvers being implemented)
- ⏳ API gateway waiting for service endpoints
- ⏳ Security audit waiting for all code completion

---

### Phase 7: Retrieve Completed Task Results

```javascript
// Step 7: Get results from completed database schema task
mcp__flow-nexus__task_results({
  taskId: "task-database-schema-003",
  format: "detailed"
})
```

**Expected Output:**
```json
{
  "task_id": "task-database-schema-003",
  "status": "completed",
  "agent": "agent-database-architect-g7h8i9",
  "duration_minutes": 30,
  "results": {
    "files_created": [
      "/database/migrations/001_create_users.sql",
      "/database/migrations/002_create_products.sql",
      "/database/migrations/003_create_orders.sql",
      "/database/schema-diagram.png",
      "/database/indexes.sql",
      "/database/seed-data.sql"
    ],
    "schema_summary": {
      "tables": 9,
      "indexes": 12,
      "foreign_keys": 8,
      "partitions": 1,
      "total_columns": 47
    },
    "quality_metrics": {
      "normalization_level": "3NF",
      "indexes_on_foreign_keys": "100%",
      "constraints_validated": true,
      "migration_tested": true
    },
    "logs": "Database schema designed with 9 tables, 12 indexes, and monthly partitioning on orders table. All foreign keys have appropriate cascade rules. Schema validated against PostgreSQL 15."
  }
}
```

---

## Complete Code Example

```javascript
// ========================================
// FLOW NEXUS TASK ORCHESTRATION
// E-Commerce Microservices Development
// ========================================

async function orchestrateEcommerceDevelopment() {
  console.log("🎯 Orchestrating E-Commerce Platform Development...\n");

  const tasks = [];

  // Phase 1: Critical Path - Authentication Service
  console.log("📋 Task 1: Authentication Service (Critical Priority)");
  tasks.push(await mcp__flow-nexus__task_orchestrate({
    task: "Build OAuth 2.0 + JWT authentication service...",
    strategy: "adaptive",
    priority: "critical",
    maxAgents: 2
  }));

  // Phase 2: Parallel - Product Catalog Service
  console.log("📋 Task 2: Product Catalog Service (Parallel Execution)");
  tasks.push(await mcp__flow-nexus__task_orchestrate({
    task: "Build GraphQL product catalog service...",
    strategy: "parallel",
    priority: "high",
    maxAgents: 2
  }));

  // Phase 3: Foundation - Database Schema
  console.log("📋 Task 3: Database Schema Design (Sequential)");
  tasks.push(await mcp__flow-nexus__task_orchestrate({
    task: "Design comprehensive PostgreSQL database schema...",
    strategy: "sequential",
    priority: "critical",
    maxAgents: 1
  }));

  // Phase 4: Integration - API Gateway
  console.log("📋 Task 4: API Gateway (Depends on Services)");
  tasks.push(await mcp__flow-nexus__task_orchestrate({
    task: "Design API Gateway with OpenAPI 3.1 specification...",
    strategy: "adaptive",
    priority: "medium",
    maxAgents: 2
  }));

  // Phase 5: Quality - Security Audit
  console.log("📋 Task 5: Security Audit (Final Quality Check)");
  tasks.push(await mcp__flow-nexus__task_orchestrate({
    task: "Perform comprehensive security audit on e-commerce platform...",
    strategy: "sequential",
    priority: "high",
    maxAgents: 1
  }));

  console.log(`\n✅ Orchestrated ${tasks.length} tasks\n`);

  // Monitor progress every 30 seconds
  const monitorInterval = setInterval(async () => {
    const status = await mcp__flow-nexus__task_status({ detailed: true });

    console.log("\n📊 Task Progress Update:");
    console.log(`   Completed: ${status.summary.completed}/${status.summary.total_tasks}`);
    console.log(`   Running: ${status.summary.running}`);
    console.log(`   Pending: ${status.summary.pending}`);

    status.tasks.forEach(task => {
      console.log(`   ${task.task_id}: ${task.status} (${task.progress_percentage}%)`);
    });

    // Stop monitoring when all tasks complete
    if (status.summary.completed === status.summary.total_tasks) {
      clearInterval(monitorInterval);
      console.log("\n🎉 All tasks completed!");

      // Retrieve all results
      for (const task of tasks) {
        const results = await mcp__flow-nexus__task_results({
          taskId: task.task_id,
          format: "detailed"
        });
        console.log(`\n📦 Results for ${task.task_id}:`);
        console.log(`   Files: ${results.results.files_created?.length || 0}`);
        console.log(`   Duration: ${results.duration_minutes} minutes`);
      }
    }
  }, 30000); // Check every 30 seconds

  return tasks;
}

// Execute orchestration
orchestrateEcommerceDevelopment()
  .then(tasks => {
    console.log("\n✅ Task orchestration initiated!");
    console.log(`   Total tasks: ${tasks.length}`);
  })
  .catch(error => {
    console.error("❌ Task orchestration failed:", error);
  });
```

---

## Outcomes and Results

### Task Execution Timeline

**Minute 0-30: Parallel Execution**
- ✅ Database schema completed (30 min)
- 🏃 Auth service in progress (45 min total)
- 🏃 Catalog service in progress (60 min total)

**Minute 30-45: Auth Service Completion**
- ✅ Auth service completed (45 min)
- 🏃 Catalog service still running (60 min total)
- ⏳ API gateway waiting

**Minute 45-60: Catalog Service Completion**
- ✅ Catalog service completed (60 min)
- 🏃 API gateway started (40 min)

**Minute 60-100: API Gateway + Security Audit**
- ✅ API gateway completed (100 min total)
- 🏃 Security audit started (90 min)

**Minute 100-190: Security Audit Completion**
- ✅ Security audit completed (190 min total)
- ✅ **All tasks complete!**

**Total Time: ~3 hours 10 minutes** (vs ~5 hours sequential)

---

## Pro Tips and Best Practices

### Tip 1: Use Priority Wisely

**Critical Priority:**
- Authentication (blocks all protected endpoints)
- Database schema (blocks all data operations)
- Deployment scripts (blocks production release)

**High Priority:**
- Core business features
- Security audits
- Integration testing

**Medium Priority:**
- Documentation
- API design
- Monitoring setup

**Low Priority:**
- Performance optimizations
- UI polish
- Analytics integration

### Tip 2: Optimize Strategy Selection

| Strategy | When to Use | Example |
|----------|-------------|---------|
| `adaptive` | Best-performing agent should execute | Complex algorithms, critical features |
| `parallel` | Tasks are completely independent | Microservices, test suites, documentation |
| `sequential` | Tasks have dependencies | Database migrations, deployment pipelines |

### Tip 3: Batch Related Tasks

```javascript
// ❌ DON'T: Orchestrate one feature at a time
await orchestrateTask("Build auth");
await orchestrateTask("Build catalog");
await orchestrateTask("Build orders");
// Result: Slow, sequential, no parallelism

// ✅ DO: Orchestrate all independent tasks together
const [auth, catalog, orders] = await Promise.all([
  orchestrateTask("Build auth"),
  orchestrateTask("Build catalog"),
  orchestrateTask("Build orders")
]);
// Result: Fast, parallel, efficient
```

### Tip 4: Monitor Task Progress Actively

```javascript
// Set up real-time monitoring
const monitor = setInterval(async () => {
  const status = await task_status({ detailed: true });

  // Check for failures
  const failures = status.tasks.filter(t => t.status === "failed");
  if (failures.length > 0) {
    console.error("⚠️ Task failures detected:", failures);
    // Retry or escalate
  }

  // Check for long-running tasks
  const longRunning = status.tasks.filter(t =>
    t.status === "running" && t.duration_minutes > 120
  );
  if (longRunning.length > 0) {
    console.warn("⏰ Long-running tasks:", longRunning);
    // Consider adding more agents
  }
}, 60000); // Check every minute
```

### Tip 5: Handle Task Dependencies

```javascript
// Method 1: Manual dependency checking
const dbTask = await orchestrateTask("Database schema");
await waitForCompletion(dbTask.task_id);
const serviceTask = await orchestrateTask("Service using DB");

// Method 2: Dependency tracking (Flow Nexus handles this)
const tasks = {
  db: await orchestrateTask("Database schema"),
  service: await orchestrateTask("Service", {
    dependencies: ["db"]  // Will auto-wait for db task
  })
};
```

---

## Common Pitfalls to Avoid

### ❌ Pitfall 1: Not Checking Task Results
```javascript
// DON'T: Assume task succeeded
await orchestrateTask("Build feature");
// Immediately deploy without checking

// DO: Validate task results
const task = await orchestrateTask("Build feature");
const results = await task_results({ taskId: task.task_id });
if (results.status !== "completed") {
  console.error("Task failed, aborting deployment");
  return;
}
```

### ❌ Pitfall 2: Over-Orchestrating Small Tasks
```javascript
// DON'T: Orchestrate trivial tasks
await orchestrateTask("Fix typo in comment");
await orchestrateTask("Update README");
// Result: Orchestration overhead > task execution time

// DO: Batch small tasks or do them directly
await orchestrateTask("Documentation updates: Fix typo + Update README");
```

### ❌ Pitfall 3: Ignoring Agent Capabilities
```javascript
// DON'T: Assign mismatched tasks
await orchestrateTask("Machine learning model", {
  maxAgents: 1  // Might assign backend developer
});
// Result: Agent lacks ML capabilities

// DO: Ensure agents have required capabilities
await orchestrateTask("Machine learning model", {
  maxAgents: 1,
  requiredCapabilities: ["tensorflow", "model-training"]
});
```

---

## Next Steps

After task orchestration:

1. **Monitor Agent Performance** → Use `agent_metrics` to identify bottlenecks
2. **Coordinate Real-Time Collaboration** → See `example-3-agent-coordination.md`
3. **Scale Swarm** → Add more agents if tasks are queued
4. **Deploy Results** → Use DevOps agent for container deployment

---

**Flow Nexus Version**: 1.5.0
**Last Updated**: 2025-11-02
**Skill**: `flow-nexus-swarm`
