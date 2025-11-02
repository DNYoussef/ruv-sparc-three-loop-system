# MCP Integration Patterns for Agents

## Overview

Model Context Protocol (MCP) enables agents to coordinate, share memory, and access external tools. This document covers common integration patterns for Claude Flow MCP, Memory MCP, and other MCP servers.

---

## Claude Flow MCP Integration

### Pattern 1: Swarm Initialization

**Use Case**: Initialize multi-agent coordination topology

```javascript
// Hierarchical topology (Coordinator → Specialists)
mcp__claude-flow__swarm_init({
  topology: "hierarchical",
  maxAgents: 10,
  strategy: "balanced"  // Distribute work evenly
})

// Mesh topology (Peer-to-peer coordination)
mcp__claude-flow__swarm_init({
  topology: "mesh",
  maxAgents: 6,
  strategy: "adaptive"  // Adjust coordination dynamically
})

// Star topology (Central coordinator)
mcp__claude-flow__swarm_init({
  topology: "star",
  maxAgents: 8,
  strategy: "specialized"  // Each agent has unique role
})
```

**When to Use**:
- ✅ Multi-agent workflows (3+ agents)
- ✅ Complex coordination (dependencies, parallel execution)
- ✅ When error recovery needed (rollback, retry)

**Integration in Agent**:
```markdown
## 🔧 MCP SERVER TOOLS I USE

**Claude Flow MCP**:
- `mcp__claude-flow__swarm_init`
  - **WHEN**: Starting multi-agent workflow (e.g., full-stack deployment)
  - **HOW**: `swarm_init({ topology: "hierarchical", maxAgents: 10 })`
```

---

### Pattern 2: Agent Spawning

**Use Case**: Spawn specialist agents for sub-tasks

```javascript
// Spawn single specialist agent
mcp__claude-flow__agent_spawn({
  type: "backend-developer",
  name: "api-builder",
  capabilities: ["nodejs", "express", "jwt-auth"]
})

// Spawn multiple agents in parallel
const agents = [
  { type: "frontend-developer", name: "ui-builder" },
  { type: "database-specialist", name: "db-designer" },
  { type: "devops-specialist", name: "docker-setup" }
];

agents.forEach(agent => {
  mcp__claude-flow__agent_spawn(agent);
});
```

**When to Use**:
- ✅ Delegating complex sub-tasks
- ✅ Parallel execution (multiple independent tasks)
- ✅ When specialist expertise needed

**Integration in Agent**:
```markdown
**Claude Flow MCP**:
- `mcp__claude-flow__agent_spawn`
  - **WHEN**: Delegating to specialist (e.g., complex algorithm → Algorithm Specialist)
  - **HOW**: `agent_spawn({ type: "algorithm-specialist", task: "Implement recommendation engine" })`
```

---

### Pattern 3: Task Orchestration

**Use Case**: High-level workflow orchestration with automatic agent selection

```javascript
// Adaptive orchestration (Claude Flow selects agents)
mcp__claude-flow__task_orchestrate({
  task: "Deploy e-commerce application to production",
  strategy: "adaptive",  // Adjust parallelism based on workload
  priority: "high",
  maxAgents: 8
})

// Sequential orchestration
mcp__claude-flow__task_orchestrate({
  task: "Database migration with zero downtime",
  strategy: "sequential",  // Must run in order
  priority: "critical"
})

// Parallel orchestration
mcp__claude-flow__task_orchestrate({
  task: "Run test suite across multiple browsers",
  strategy: "parallel",  // Run simultaneously
  priority: "medium",
  maxAgents: 5
})
```

**When to Use**:
- ✅ Complex workflows (5+ steps)
- ✅ When automatic agent selection preferred
- ✅ When Claude Flow should handle coordination

**Integration in Agent**:
```markdown
**Claude Flow MCP**:
- `mcp__claude-flow__task_orchestrate`
  - **WHEN**: High-level workflow with automatic agent coordination
  - **HOW**: `task_orchestrate({ task: "Deploy app", strategy: "adaptive", priority: "high" })`
```

---

### Pattern 4: Memory Storage & Retrieval

**Use Case**: Share data between agents, persist workflow state

```javascript
// Store campaign strategy (available to all agents)
mcp__claude-flow__memory_store({
  key: "marketing/campaign-2024-q4/strategy",
  value: {
    budget: 100000,
    target_roas: 4.0,
    segments: [
      { name: "repeat-customers", budget_pct: 0.40 },
      { name: "new-customers", budget_pct: 0.35 },
      { name: "gift-buyers", budget_pct: 0.25 }
    ],
    channels: ["google-ads", "facebook", "email"]
  },
  ttl: 2592000  // 30 days (long-term)
})

// Retrieve campaign strategy (another agent)
const strategy = mcp__claude-flow__memory_retrieve({
  key: "marketing/campaign-2024-q4/strategy"
})

// Namespace convention: {domain}/{project-id}/{data-type}
// Examples:
// - "marketing/campaign-2024-q4/strategy"
// - "devops/deployment-prod-v2.5/state"
// - "backend-dev/api-v3/schema-design"
```

**Namespace Conventions**:
```
{agent-role}/{project-id}/{data-type}

Components:
- agent-role: marketing, devops, backend-dev, frontend-dev, etc.
- project-id: Unique identifier (campaign ID, deployment ID, feature ID)
- data-type: strategy, state, schema, metrics, etc.

TTL Guidelines:
- Short-term (24h): Daily metrics, temporary state
- Mid-term (7d): Campaign data, feature specs
- Long-term (30d+): Historical benchmarks, templates
```

**When to Use**:
- ✅ Cross-agent data sharing (API contracts, feature specs)
- ✅ Workflow state persistence (deployment state, resource IDs)
- ✅ Historical data storage (benchmarks, templates)

**Integration in Agent**:
```markdown
**Claude Flow MCP**:
- `mcp__claude-flow__memory_store`
  - **WHEN**: Storing results for other agents or future use
  - **HOW**: Namespace: `{agent-role}/{project-id}/{data-type}`
  - **Example**: `marketing/campaign-123/audience-analysis`

- `mcp__claude-flow__memory_retrieve`
  - **WHEN**: Loading context from other agents or previous sessions
  - **HOW**: `memory_retrieve({ key: "marketing/campaign-123/strategy" })`
```

---

### Pattern 5: Agent Status Monitoring

**Use Case**: Monitor health of spawned agents during orchestration

```javascript
// Check overall swarm status
const status = mcp__claude-flow__swarm_status({
  verbose: true  // Include detailed agent information
})

// Example output:
// {
//   swarm_id: "swarm-abc123",
//   topology: "hierarchical",
//   agents: [
//     { id: "agent-1", type: "backend-dev", status: "running", progress: 0.65 },
//     { id: "agent-2", type: "frontend-dev", status: "completed", progress: 1.0 },
//     { id: "agent-3", type: "database-specialist", status: "failed", error: "Connection timeout" }
//   ],
//   health: "degraded"  // One agent failed
// }

// List all agents
const agents = mcp__claude-flow__agent_list({
  filter: "active"  // Only active agents
})

// Get metrics for specific agent
const metrics = mcp__claude-flow__agent_metrics({
  agentId: "agent-1",
  metric: "all"  // cpu, memory, tasks, performance
})
```

**When to Use**:
- ✅ Coordinator agents monitoring specialists
- ✅ Health checks during long-running workflows
- ✅ Error detection and recovery

**Integration in Agent**:
```markdown
**Claude Flow MCP**:
- `mcp__claude-flow__swarm_status`
  - **WHEN**: Monitoring deployment progress, checking agent health
  - **HOW**: Poll every 30s during workflow, check `status` field

- `mcp__claude-flow__agent_metrics`
  - **WHEN**: Detecting bottlenecks, optimizing performance
  - **HOW**: `agent_metrics({ agentId: "agent-1", metric: "performance" })`
```

---

## Memory MCP Integration

### Pattern 6: Persistent Cross-Session Storage

**Use Case**: Store data that persists across sessions, searchable with semantic similarity

```javascript
// Store audience persona (persistent, long-term)
mcp__memory-mcp__memory_store({
  text: `
    Audience Segment: "High-Value Repeat Customers"
    Demographics: Women 25-44, household income $75k+, suburban
    Psychographics: Value quality over price, family-oriented, time-constrained
    Behavior: 3+ purchases in past year, average LTV $450, CAC $85
    Motivations: Convenience, trusted brands, exclusive perks
    Messaging: Emphasize loyalty rewards, exclusive offers, time-saving benefits
    Channels: Email (open rate 24%), Facebook (ROAS 4.2:1), Google retargeting
  `,
  metadata: {
    key: "marketing/personas/high-value-repeat-customers",
    namespace: "agents/marketing",
    layer: "long-term",  // Persist 30+ days
    category: "audience-research",
    project: "e-commerce-marketing",
    agent: "marketing-specialist",
    timestamp: new Date().toISOString()
  }
})

// Retrieve with semantic search (find similar personas)
const results = mcp__memory-mcp__vector_search({
  query: "audience persona for loyal customers with high lifetime value",
  limit: 5
})

// Example results:
// [
//   { text: "Audience Segment: High-Value Repeat Customers...", similarity: 0.92 },
//   { text: "Audience Segment: VIP Club Members...", similarity: 0.84 },
//   ...
// ]
```

**Tagging Protocol** (Required for ALL writes):
```javascript
metadata: {
  key: "{namespace}/{specific-id}",           // Unique identifier
  namespace: "agents/{agent-category}",       // Organizational hierarchy
  layer: "short-term|mid-term|long-term",    // Retention period (24h, 7d, 30d+)
  category: "{data-category}",               // Data type (analysis, design, metrics)
  project: "{project-name}",                 // Project/feature identifier
  agent: "{agent-name}",                     // Agent that created data
  timestamp: new Date().toISOString(),       // ISO timestamp
  intent: "{purpose}"                        // Why this was stored (optional)
}
```

**When to Use**:
- ✅ Long-term knowledge storage (personas, templates, benchmarks)
- ✅ Cross-session context (retrieve from previous sessions)
- ✅ Semantic search (find similar past work)

**Integration in Agent**:
```markdown
**Memory MCP**:
- `mcp__memory-mcp__memory_store`
  - **WHEN**: Storing insights for future use (personas, templates, benchmarks)
  - **HOW**: Auto-tagged with WHO (agent), WHEN (timestamp), PROJECT, WHY (intent)
  - **Example**: Store campaign template for reuse in future campaigns

- `mcp__memory-mcp__vector_search`
  - **WHEN**: Finding similar past work (campaigns, features, solutions)
  - **HOW**: `vector_search({ query: "campaign strategy for Q4 holiday", limit: 10 })`
```

---

### Pattern 7: Mode-Aware Context Adaptation

**Use Case**: Memory MCP automatically adapts retention based on interaction mode

```javascript
// Execution mode (short-term, 24h retention)
mcp__memory-mcp__memory_store({
  text: "Deployment completed: ecommerce-app v2.5.0, all health checks passed",
  metadata: {
    // Mode auto-detected as "execution" based on content patterns
    // Automatically assigned layer: "short-term" (24h retention)
  }
})

// Planning mode (mid-term, 7d retention)
mcp__memory-mcp__memory_store({
  text: "Campaign strategy for Q4: 3 audience segments, $100k budget allocation...",
  metadata: {
    // Mode auto-detected as "planning"
    // Automatically assigned layer: "mid-term" (7d retention)
  }
})

// Brainstorming mode (short-term, 24h retention)
mcp__memory-mcp__memory_store({
  text: "Ideas for homepage redesign: hero image carousel, testimonial section...",
  metadata: {
    // Mode auto-detected as "brainstorming"
    // Automatically assigned layer: "short-term" (24h retention)
  }
})
```

**Mode Detection Patterns** (29 patterns, automatic):
- **Execution**: "deployed", "completed", "executed", "ran", "tested"
- **Planning**: "strategy", "plan", "roadmap", "approach", "proposal"
- **Brainstorming**: "ideas", "possibilities", "what if", "brainstorm", "explore"

**When to Use**:
- ✅ Let Memory MCP handle retention automatically
- ✅ For consistent retention policies across agents
- ✅ When mode is clear from content

**Integration in Agent**:
```markdown
**Memory MCP**:
- Mode-aware context adaptation (automatic)
  - **HOW**: Memory MCP detects mode from content (29 patterns)
  - **RESULT**: Automatic layer assignment (short/mid/long-term)
  - **BENEFIT**: No manual retention management needed
```

---

## Connascence Analyzer Integration (Code Quality Agents)

### Pattern 8: Code Quality Analysis

**Use Case**: Detect code violations, coupling issues, NASA compliance

```javascript
// Analyze single file for connascence violations
const analysis = mcp__connascence-analyzer__analyze_file({
  filePath: "backend/services/userService.ts"
})

// Example output:
// {
//   violations: [
//     {
//       type: "God Object",
//       severity: "high",
//       location: "userService.ts:15",
//       message: "Class has 26 methods (threshold: 15)",
//       suggestion: "Split into UserRepository, UserValidator, UserService"
//     },
//     {
//       type: "Parameter Bomb (CoP)",
//       severity: "critical",
//       location: "userService.ts:45",
//       message: "Function has 14 parameters (NASA limit: 6)",
//       suggestion: "Use options object pattern"
//     },
//     {
//       type: "Cyclomatic Complexity",
//       severity: "medium",
//       location: "userService.ts:89",
//       message: "Function complexity: 13 (threshold: 10)",
//       suggestion: "Extract conditional logic into separate functions"
//     }
//   ],
//   analysisTime: 0.018,
//   totalViolations: 7
// }

// Analyze entire workspace
const workspaceAnalysis = mcp__connascence-analyzer__analyze_workspace({
  path: "backend/",
  recursive: true
})
```

**Detection Capabilities**:
1. God Objects (>15 methods)
2. Parameter Bombs (>6 params, NASA limit)
3. Cyclomatic Complexity (>10)
4. Deep Nesting (>4 levels, NASA limit)
5. Long Functions (>50 lines)
6. Magic Literals (hardcoded values)
7. Security violations (SQL injection, XSS)

**When to Use**:
- ✅ Code quality agents only (coder, reviewer, tester, code-analyzer)
- ✅ Before code review (catch violations early)
- ✅ CI/CD pipeline (fail build on critical violations)

**Agent Access** (14 agents):
- coder, reviewer, tester, code-analyzer
- functionality-audit, theater-detection-audit, production-validator
- sparc-coder, analyst, backend-dev, mobile-dev
- ml-developer, base-template-generator, code-review-swarm

**Integration in Agent**:
```markdown
**Connascence Analyzer** (Code Quality Agents Only):
- `mcp__connascence-analyzer__analyze_file`
  - **WHEN**: Before finalizing code, during code review
  - **HOW**: `analyze_file({ filePath: "path/to/file.ts" })`
  - **ACTION**: Fix violations before committing

- `mcp__connascence-analyzer__analyze_workspace`
  - **WHEN**: Full project audit, CI/CD pipeline
  - **HOW**: `analyze_workspace({ path: "backend/", recursive: true })`
```

---

## Flow-Nexus MCP Integration (Optional Advanced Features)

### Pattern 9: Cloud Sandbox Execution

**Use Case**: Execute code in isolated cloud environments (E2B sandboxes)

```javascript
// Create sandbox for testing
const sandbox = mcp__flow-nexus__sandbox_create({
  template: "nodejs",  // nodejs, python, react, nextjs
  env_vars: {
    DATABASE_URL: "postgresql://localhost/testdb",
    JWT_SECRET: "test-secret-key"
  },
  timeout: 3600,  // 1 hour
  install_packages: ["express", "prisma", "jsonwebtoken"]
})

// Execute code in sandbox
const result = mcp__flow-nexus__sandbox_execute({
  sandbox_id: sandbox.id,
  code: `
    const express = require('express');
    const app = express();
    app.get('/health', (req, res) => res.json({ status: 'ok' }));
    app.listen(3000, () => console.log('Server running'));
  `,
  timeout: 60
})

// Upload test data to sandbox
mcp__flow-nexus__sandbox_upload({
  sandbox_id: sandbox.id,
  file_path: "/app/test-data.json",
  content: JSON.stringify({ users: [...] })
})

// Clean up sandbox
mcp__flow-nexus__sandbox_delete({
  sandbox_id: sandbox.id
})
```

**When to Use**:
- ✅ Testing agents (validate functionality in isolation)
- ✅ Security agents (analyze suspicious code safely)
- ✅ Deployment agents (test before production)

---

### Pattern 10: Real-Time Execution Monitoring

**Use Case**: Monitor agent execution in real-time with streaming updates

```javascript
// Subscribe to execution stream
mcp__flow-nexus__execution_stream_subscribe({
  stream_type: "claude-flow-swarm",
  deployment_id: "ecommerce-app-2024-11-02"
})

// Monitor execution progress (callback pattern)
onExecutionUpdate((event) => {
  console.log(`Agent ${event.agent_id}: ${event.status}`);
  console.log(`Progress: ${event.progress * 100}%`);

  if (event.status === "completed") {
    console.log(`Agent completed in ${event.duration}ms`);
  } else if (event.status === "failed") {
    console.error(`Agent failed: ${event.error}`);
    triggerRollback();
  }
});
```

**When to Use**:
- ✅ Coordinator agents monitoring specialists
- ✅ Long-running workflows (deployments, data processing)
- ✅ User-facing dashboards (show progress)

---

## Integration Best Practices

### 1. Namespace Conventions

**Claude Flow MCP**:
```
Format: {agent-role}/{project-id}/{data-type}

Examples:
- marketing/campaign-2024-q4/strategy
- devops/deployment-prod-v2.5/state
- backend-dev/api-v3/schema-design
```

**Memory MCP**:
```
Format: agents/{agent-category}/{specific-id}

Examples:
- agents/marketing/personas/high-value-customers
- agents/devops/templates/docker-compose-nodejs
- agents/backend-dev/patterns/rest-api-auth
```

### 2. Error Handling

```javascript
try {
  const result = mcp__claude-flow__agent_spawn({
    type: "backend-developer",
    task: "Implement REST API"
  });
} catch (error) {
  console.error("Failed to spawn agent:", error);

  // Fallback: Implement directly (if within capability)
  if (canImplementDirectly()) {
    implementDirectly();
  } else {
    // Escalate to coordinator
    escalateToCoordinator(error);
  }
}
```

### 3. Performance Optimization

**Batch Operations**:
```javascript
// ❌ BAD: Sequential spawning
agents.forEach(agent => {
  mcp__claude-flow__agent_spawn(agent);  // Slow
});

// ✅ GOOD: Parallel spawning
Promise.all(
  agents.map(agent => mcp__claude-flow__agent_spawn(agent))
);
```

**Memory Caching**:
```javascript
// Cache frequently accessed data
let campaignStrategy = null;

function getCampaignStrategy() {
  if (!campaignStrategy) {
    campaignStrategy = mcp__claude-flow__memory_retrieve({
      key: "marketing/campaign-2024-q4/strategy"
    });
  }
  return campaignStrategy;
}
```

### 4. Testing MCP Integration

```javascript
// Mock MCP calls for testing
const mockMcpTools = {
  agent_spawn: jest.fn(() => ({ id: "agent-123", status: "running" })),
  memory_store: jest.fn(() => ({ success: true })),
  memory_retrieve: jest.fn(() => ({ data: "test-data" }))
};

// Test agent behavior with mocked MCP
const agent = new MarketingSpecialist({ mcpTools: mockMcpTools });
agent.execute("Analyze campaign");

expect(mockMcpTools.memory_retrieve).toHaveBeenCalledWith({
  key: "marketing/campaign-2024-q4/strategy"
});
```

---

## Summary

**Key MCP Tools**:
1. **Claude Flow MCP**: Swarm coordination, agent spawning, memory storage
2. **Memory MCP**: Persistent cross-session storage, semantic search
3. **Connascence Analyzer**: Code quality checks (code quality agents only)
4. **Flow-Nexus**: Cloud sandboxes, real-time monitoring (optional)

**Integration Patterns**:
- ✅ Namespace conventions ({agent-role}/{project}/{data-type})
- ✅ Tagging protocol (WHO/WHEN/PROJECT/WHY for Memory MCP)
- ✅ Error handling (try-catch, fallback, escalation)
- ✅ Performance optimization (batch operations, caching)
- ✅ Testing (mock MCP tools)

**Agent-Specific Integration**:
- **Specialists**: Minimal MCP usage (memory storage for results)
- **Coordinators**: Heavy MCP usage (swarm init, agent spawning, status monitoring)
- **Hybrids**: Moderate MCP usage (memory + occasional delegation)

Use these patterns to build agents with robust MCP integration for production-ready coordination and collaboration.
