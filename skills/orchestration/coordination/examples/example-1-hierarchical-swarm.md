# Example 1: Hierarchical Swarm Coordination

## Scenario Description

You need to build a full-stack e-commerce application with multiple microservices. The project requires coordinating 12 agents across frontend, backend, database, testing, and DevOps domains. A hierarchical swarm topology is ideal for this complex, multi-layered architecture.

**Project Components**:
- Frontend (React + Next.js)
- Backend APIs (Node.js + Express)
- Database layer (PostgreSQL + Redis)
- Payment service (Stripe integration)
- Search service (Elasticsearch)
- Testing suite (Jest + Cypress)
- CI/CD pipeline (GitHub Actions)
- Infrastructure (Docker + Kubernetes)

**Why Hierarchical?**
- Clear command chain for complex projects
- Specialized teams (frontend, backend, infrastructure)
- Coordinator agents manage sub-teams
- Scales well for 8+ agents

---

## Step-by-Step Walkthrough

### Step 1: Initialize Hierarchical Swarm

```bash
# Initialize hierarchical topology with coordinator at the top
npx claude-flow@alpha swarm init hierarchical --max-agents 12
```

**Expected Output**:
```json
{
  "swarmId": "swarm-ecommerce-123",
  "topology": "hierarchical",
  "maxAgents": 12,
  "status": "initialized",
  "rootCoordinator": "coordinator-root-001"
}
```

**What Happens**:
- Creates hierarchical tree structure
- Spawns root coordinator agent
- Prepares memory for agent registry
- Sets up coordination hooks

---

### Step 2: Spawn Coordinator Agents for Each Domain

```javascript
// In Claude Code - spawn coordinators in parallel
[Single Message - Parallel Coordinator Spawning]:

// Root coordinator oversees all domains
Task("Root Coordinator",
  "Orchestrate e-commerce platform development across 4 domains: frontend, backend, infrastructure, testing. Coordinate with domain coordinators. Track overall progress and resolve cross-domain dependencies.",
  "coordinator")

// Frontend domain coordinator
Task("Frontend Coordinator",
  "Manage frontend team (React developer, UI specialist, accessibility specialist). Coordinate component library, pages, and state management. Report to root coordinator.",
  "coordinator")

// Backend domain coordinator
Task("Backend Coordinator",
  "Manage backend team (API developer, database specialist, payment integration specialist). Coordinate microservices architecture. Report to root coordinator.",
  "coordinator")

// Infrastructure domain coordinator
Task("Infrastructure Coordinator",
  "Manage DevOps team (Docker specialist, K8s specialist, CI/CD engineer). Coordinate deployment pipeline and monitoring. Report to root coordinator.",
  "coordinator")
```

**Coordination Protocol**:
Each coordinator runs hooks:
```bash
# Before coordinating
npx claude-flow@alpha hooks pre-task --description "Coordinate [domain] team"
npx claude-flow@alpha hooks session-restore --session-id "swarm-ecommerce-123"

# During coordination
npx claude-flow@alpha hooks notify --message "Domain status update: [status]"
npx claude-flow@alpha hooks post-edit --file "coordination/[domain]-status.json"

# After coordination
npx claude-flow@alpha hooks post-task --task-id "coord-[domain]"
```

---

### Step 3: Spawn Specialist Agents Under Coordinators

```javascript
// In Claude Code - spawn specialists under frontend coordinator
[Single Message - Frontend Team]:

Task("React Developer",
  "Build Next.js 14 App Router application. Implement product catalog, cart, checkout pages. Use Server Components and Server Actions. Report to Frontend Coordinator.",
  "coder")

Task("UI Component Specialist",
  "Create reusable component library with Tailwind CSS and shadcn/ui. Design system tokens, theming, responsive layouts. Report to Frontend Coordinator.",
  "coder")

Task("Accessibility Specialist",
  "Ensure WCAG 2.1 AA compliance. Implement keyboard navigation, screen reader support, ARIA labels. Audit with axe-core. Report to Frontend Coordinator.",
  "analyst")

// In parallel - spawn backend team
[Single Message - Backend Team]:

Task("API Developer",
  "Implement REST APIs for products, cart, orders, users. Express.js with TypeScript. JWT authentication. Report to Backend Coordinator.",
  "coder")

Task("Database Specialist",
  "Design PostgreSQL schema for e-commerce (products, orders, users, inventory). Write migrations. Optimize queries. Report to Backend Coordinator.",
  "coder")

Task("Payment Integration Specialist",
  "Integrate Stripe payment processing. Handle webhooks, subscription billing, refunds. PCI compliance. Report to Backend Coordinator.",
  "coder")

// In parallel - spawn infrastructure team
[Single Message - Infrastructure Team]:

Task("Docker Specialist",
  "Create multi-stage Dockerfiles for all services. Docker Compose for local dev. Optimize image sizes. Report to Infrastructure Coordinator.",
  "coder")

Task("Kubernetes Specialist",
  "Design K8s manifests for production deployment. Helm charts. Service mesh with Istio. Auto-scaling policies. Report to Infrastructure Coordinator.",
  "coder")

Task("CI/CD Engineer",
  "Build GitHub Actions pipelines for testing, building, deploying. Environment promotion (dev→staging→prod). Report to Infrastructure Coordinator.",
  "coder")

// In parallel - spawn testing team (reports to root)
[Single Message - Testing Team]:

Task("Unit Test Engineer",
  "Write Jest unit tests for all services. 90%+ coverage. Mock external dependencies. Report to Root Coordinator.",
  "analyst")

Task("E2E Test Engineer",
  "Create Cypress E2E tests for critical user flows (browse, add to cart, checkout). Visual regression testing. Report to Root Coordinator.",
  "analyst")

Task("Performance Test Engineer",
  "Load testing with k6. API performance benchmarks. Database query optimization. Report to Root Coordinator.",
  "optimizer")
```

---

### Step 4: Coordination Flow - Hierarchical Communication

**Communication Hierarchy**:
```
Root Coordinator
├── Frontend Coordinator
│   ├── React Developer
│   ├── UI Component Specialist
│   └── Accessibility Specialist
├── Backend Coordinator
│   ├── API Developer
│   ├── Database Specialist
│   └── Payment Integration Specialist
├── Infrastructure Coordinator
│   ├── Docker Specialist
│   ├── Kubernetes Specialist
│   └── CI/CD Engineer
└── Testing Team (direct reports)
    ├── Unit Test Engineer
    ├── E2E Test Engineer
    └── Performance Test Engineer
```

**Status Reporting Pattern**:
```javascript
// Specialist → Domain Coordinator (every 30 minutes)
npx claude-flow@alpha hooks notify --message "React Developer: Product catalog 80% complete. Blocked on API endpoints."

// Domain Coordinator → Root Coordinator (hourly)
npx claude-flow@alpha hooks notify --message "Frontend Coordinator: Team 75% complete. Waiting on backend APIs."

// Root Coordinator → User (every 2 hours)
npx claude-flow@alpha hooks session-status --export-summary true
```

---

### Step 5: Cross-Domain Dependency Resolution

**Scenario**: React Developer needs API endpoints from Backend Team

**Resolution Flow**:
1. React Developer reports blockage to Frontend Coordinator
2. Frontend Coordinator escalates to Root Coordinator
3. Root Coordinator queries Backend Coordinator for API status
4. Backend Coordinator prioritizes API endpoints
5. API Developer completes endpoints
6. Root Coordinator notifies Frontend Coordinator
7. Frontend Coordinator unblocks React Developer

**Implementation with Memory MCP**:
```javascript
// React Developer stores blockage
mcp__memory__store({
  key: "swarm/blockages/react-dev-001",
  value: JSON.stringify({
    agent: "React Developer",
    blocker: "Missing API endpoints: GET /api/products, POST /api/cart",
    blockedSince: new Date().toISOString(),
    coordinator: "Frontend Coordinator"
  }),
  metadata: {
    tags: ["WHO:react-developer", "WHEN:" + Date.now(), "PROJECT:ecommerce", "WHY:blockage"],
    retention: "short-term"
  }
})

// Root Coordinator searches for all blockages
mcp__memory__vector_search({
  query: "agent blockages in e-commerce project",
  mode: "execution",
  topK: 10
})

// API Developer resolves and updates
mcp__memory__store({
  key: "swarm/resolutions/api-endpoints-001",
  value: JSON.stringify({
    resolved: ["GET /api/products", "POST /api/cart"],
    agent: "API Developer",
    resolvedAt: new Date().toISOString(),
    unblocks: ["React Developer"]
  }),
  metadata: {
    tags: ["WHO:api-developer", "WHEN:" + Date.now(), "PROJECT:ecommerce", "WHY:resolution"],
    retention: "mid-term"
  }
})
```

---

### Step 6: Monitor Swarm Progress

```bash
# Check overall swarm status
npx claude-flow@alpha swarm status

# Monitor real-time updates (10 seconds)
npx claude-flow@alpha swarm monitor --duration 10

# Get metrics for specific agent
npx claude-flow@alpha agent metrics --agent-id "react-developer-001"

# Export session summary
npx claude-flow@alpha hooks session-end --export-metrics true
```

---

## Expected Outcomes

### Successful Hierarchical Coordination Results:

1. **Clear Chain of Command**
   - Each agent knows their coordinator
   - Coordinators manage 3-4 specialists each
   - Root coordinator has global visibility

2. **Efficient Blockage Resolution**
   - Average resolution time: 15-30 minutes
   - Cross-domain dependencies tracked in memory
   - Escalation path defined and followed

3. **Parallel Execution**
   - 12 agents work concurrently
   - Domain coordinators prevent conflicts
   - 2.8-4.4x speed improvement vs sequential

4. **Quality Metrics**
   - 90%+ test coverage (enforced by testing team)
   - 0 critical security vulnerabilities (analyst reviews)
   - WCAG AA compliance (accessibility specialist)

5. **Deployment Success**
   - CI/CD pipeline fully automated
   - K8s manifests production-ready
   - Docker images optimized (<200MB)

---

## Code Examples

### Example: Frontend Coordinator Managing Dependencies

```javascript
// Frontend Coordinator checks if backend APIs are ready
const backendStatus = await mcp__memory__vector_search({
  query: "backend API endpoints status for products and cart",
  mode: "execution",
  topK: 5
});

if (backendStatus.results.some(r => r.content.includes("completed"))) {
  // Unblock React Developer
  await Task("React Developer",
    "Backend APIs are ready. Implement product catalog and cart integration with real endpoints.",
    "coder");

  // Notify root coordinator
  await hooks.notify({
    message: "Frontend Coordinator: React Developer unblocked. Product catalog integration starting."
  });
} else {
  // Escalate to root coordinator
  await hooks.notify({
    message: "Frontend Coordinator: BLOCKED - Waiting on backend APIs. Estimated delay: 2 hours."
  });
}
```

### Example: Root Coordinator Orchestrating Cross-Domain Integration

```javascript
// Root Coordinator checks completion status across all domains
const domainStatus = await Promise.all([
  mcp__memory__vector_search({ query: "frontend domain completion percentage", mode: "execution" }),
  mcp__memory__vector_search({ query: "backend domain completion percentage", mode: "execution" }),
  mcp__memory__vector_search({ query: "infrastructure domain completion percentage", mode: "execution" }),
  mcp__memory__vector_search({ query: "testing domain completion percentage", mode: "execution" })
]);

const [frontend, backend, infra, testing] = domainStatus.map(d =>
  parseInt(d.results[0]?.content.match(/(\d+)%/)?.[1] || "0")
);

if (frontend >= 90 && backend >= 90 && infra >= 90 && testing >= 80) {
  // Trigger integration phase
  await Task("Integration Coordinator",
    "All domains 90%+ complete. Begin integration testing and production deployment. Coordinate final QA sweep.",
    "coordinator");

  // Export final metrics
  await hooks.sessionEnd({ exportMetrics: true });
}
```

---

## Tips and Best Practices

### 1. Coordinator-to-Specialist Ratio
**Ideal**: 1 coordinator manages 3-5 specialists
- **Too few specialists** (1-2): Coordinator overhead not justified
- **Too many specialists** (6+): Coordinator becomes bottleneck

### 2. Memory MCP for Cross-Domain Communication
**Always use tagged memory** for coordination:
```javascript
// ✅ GOOD: Tagged memory with WHO/WHEN/PROJECT/WHY
mcp__memory__store({
  key: "swarm/status/frontend-team",
  value: JSON.stringify({ completion: "85%", blockers: [] }),
  metadata: {
    tags: [
      "WHO:frontend-coordinator",
      "WHEN:" + Date.now(),
      "PROJECT:ecommerce",
      "WHY:status-update"
    ],
    retention: "short-term"
  }
})

// ❌ BAD: Untagged memory (hard to search/filter)
mcp__memory__store({
  key: "status",
  value: "85%"
})
```

### 3. Escalation Protocol
**Define clear escalation paths**:
- Specialist tries to resolve (15 minutes)
- Escalate to domain coordinator (15 minutes)
- Coordinator consults memory for similar issues
- If unresolved, escalate to root coordinator
- Root coordinator has cross-domain visibility

### 4. Regular Status Updates
**Establish heartbeat schedule**:
- Specialists → Coordinators: Every 30 minutes
- Coordinators → Root: Every hour
- Root → User: Every 2 hours or on milestones

### 5. Avoid Coordinator Overload
**Warning signs**:
- Coordinator managing 8+ agents
- Response time >1 hour for escalations
- Coordinators doing specialist work

**Solution**: Add mid-level coordinators or split domains

### 6. Use Hooks for Automation
**Automate coordination tasks**:
```bash
# Auto-notify on completion
npx claude-flow@alpha hooks post-task --auto-notify true

# Auto-restore context for new agents
npx claude-flow@alpha hooks session-restore --auto-load true

# Auto-format deliverables
npx claude-flow@alpha hooks post-edit --auto-format true
```

### 7. Testing Team as Direct Reports
**Why testing reports to root**:
- Tests validate work across all domains
- Testing team needs visibility into all code
- Prevents domain coordinators from skipping tests

### 8. Document Coordination Decisions
**Store architecture decisions in memory**:
```javascript
mcp__memory__store({
  key: "swarm/decisions/arch-001",
  value: JSON.stringify({
    decision: "Use hierarchical topology for 12-agent e-commerce project",
    reasoning: "Clear chain of command, specialized domains, scales well",
    alternatives: ["mesh (too chaotic)", "star (single point of failure)"],
    decidedBy: "Root Coordinator",
    date: "2025-11-02"
  }),
  metadata: {
    tags: ["WHO:root-coordinator", "WHEN:" + Date.now(), "PROJECT:ecommerce", "WHY:architecture"],
    retention: "long-term"
  }
})
```

---

## Common Pitfalls to Avoid

1. **Micromanagement**: Don't have coordinators do specialist work
2. **Communication Silos**: Ensure cross-domain visibility via memory
3. **Unclear Ownership**: Every file/task has ONE assigned agent
4. **Skipped Hooks**: ALWAYS run pre-task/post-task hooks for coordination
5. **Ignoring Blockages**: Address blockages within 30 minutes
6. **Over-coordination**: Not every decision needs escalation
7. **No Memory Persistence**: Store decisions/status in Memory MCP
8. **Sequential Execution**: Spawn all agents in parallel in single message

---

## Next Steps

After mastering hierarchical coordination:
1. Try **mesh topology** for peer-to-peer collaboration (Example 2)
2. Explore **adaptive scaling** for dynamic agent management (Example 3)
3. Combine topologies (hierarchical for planning, mesh for execution)
4. Integrate with **audit-pipeline** for continuous quality monitoring
