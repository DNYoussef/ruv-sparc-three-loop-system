# Example 3: Agent Coordination - Real-Time Collaboration Patterns

## Scenario Overview

Your e-commerce platform development is in full swing with 6 agents working on different microservices. However, you're encountering integration challenges:

**Problems to Solve:**
1. Backend Auth Specialist changed JWT payload structure, breaking Catalog Service
2. Database Architect added new indexes, but services aren't using them
3. Security Specialist found OWASP violations, but developers don't see the report
4. DevOps Engineer deployed to staging, but API Gateway config is outdated
5. Multiple agents modifying the same files, causing merge conflicts

**Solution:** Use Flow Nexus coordination patterns to enable real-time agent collaboration, shared memory, and conflict resolution.

**Coordination Goals:**
- Shared memory for API contracts and schemas
- Real-time notifications for breaking changes
- Agent-to-agent communication for dependencies
- Performance tracking to identify bottlenecks
- Automated conflict resolution

---

## Step-by-Step Walkthrough

### Phase 1: Monitor Agent Performance Metrics

**Why Metrics First?** Identify which agents are overloaded or underutilized before coordinating work.

```javascript
// Step 1: Get overall agent performance metrics
mcp__flow-nexus__agent_metrics({
  metric: "all"
})
```

**Expected Output:**
```json
{
  "swarm_id": "swarm-ecommerce-2025",
  "metrics": {
    "overall": {
      "total_agents": 6,
      "active_agents": 6,
      "idle_agents": 0,
      "busy_agents": 4,
      "average_cpu_percent": 67.3,
      "average_memory_mb": 512.8,
      "total_tasks_completed": 8,
      "average_task_duration_minutes": 42.5
    },
    "by_agent": [
      {
        "agent_id": "agent-backend-auth-specialist-a1b2c3",
        "name": "backend-auth-specialist",
        "type": "coder",
        "status": "busy",
        "cpu_percent": 85.2,
        "memory_mb": 768,
        "tasks_completed": 2,
        "average_task_duration_minutes": 45,
        "current_task": "task-auth-service-001",
        "performance_score": 0.92
      },
      {
        "agent_id": "agent-backend-catalog-specialist-d4e5f6",
        "name": "backend-catalog-specialist",
        "type": "coder",
        "status": "busy",
        "cpu_percent": 72.1,
        "memory_mb": 612,
        "tasks_completed": 1,
        "average_task_duration_minutes": 60,
        "current_task": "task-catalog-service-002",
        "performance_score": 0.85
      },
      {
        "agent_id": "agent-database-architect-g7h8i9",
        "name": "database-architect",
        "type": "analyst",
        "status": "idle",
        "cpu_percent": 12.5,
        "memory_mb": 256,
        "tasks_completed": 3,
        "average_task_duration_minutes": 30,
        "current_task": null,
        "performance_score": 0.98
      },
      {
        "agent_id": "agent-security-specialist-m4n5o6",
        "name": "security-specialist",
        "type": "analyst",
        "status": "busy",
        "cpu_percent": 65.8,
        "memory_mb": 448,
        "tasks_completed": 1,
        "average_task_duration_minutes": 90,
        "current_task": "task-security-audit-005",
        "performance_score": 0.88
      },
      {
        "agent_id": "agent-devops-engineer-p7q8r9",
        "name": "devops-engineer",
        "type": "optimizer",
        "status": "busy",
        "cpu_percent": 58.3,
        "memory_mb": 512,
        "tasks_completed": 1,
        "average_task_duration_minutes": 35,
        "current_task": "task-docker-deployment-006",
        "performance_score": 0.90
      },
      {
        "agent_id": "agent-api-designer-j1k2l3",
        "name": "api-designer",
        "type": "researcher",
        "status": "idle",
        "cpu_percent": 8.1,
        "memory_mb": 192,
        "tasks_completed": 0,
        "average_task_duration_minutes": null,
        "current_task": null,
        "performance_score": null
      }
    ]
  },
  "timestamp": "2025-11-02T12:30:00Z"
}
```

**Performance Insights:**
- 🔴 **Bottleneck**: Backend Auth Specialist at 85.2% CPU (overloaded)
- 🟢 **Underutilized**: Database Architect at 12.5% CPU (idle)
- 🟢 **Underutilized**: API Designer at 8.1% CPU (idle, no tasks assigned)
- 🟡 **Optimal**: Catalog Specialist at 72.1% CPU (healthy workload)
- ✅ **High Performance**: Database Architect score 0.98 (fastest task completion)

**Action Items:**
1. Offload work from Auth Specialist to Database Architect
2. Assign API Designer to documentation tasks
3. Monitor Security Specialist (high CPU, long task duration)

---

### Phase 2: Get Specific Agent Metrics (Deep Dive)

```javascript
// Step 2: Drill down into Auth Specialist performance
mcp__flow-nexus__agent_metrics({
  agentId: "agent-backend-auth-specialist-a1b2c3",
  metric: "performance"
})
```

**Expected Output:**
```json
{
  "agent_id": "agent-backend-auth-specialist-a1b2c3",
  "name": "backend-auth-specialist",
  "type": "coder",
  "performance_metrics": {
    "tasks_completed": 2,
    "tasks_failed": 0,
    "average_task_duration_minutes": 45,
    "fastest_task_duration_minutes": 30,
    "slowest_task_duration_minutes": 60,
    "success_rate": 1.0,
    "average_code_quality_score": 0.92,
    "test_coverage_average": 0.95,
    "current_workload": {
      "task_id": "task-auth-service-001",
      "started_at": "2025-11-02T11:00:15Z",
      "elapsed_minutes": 90,
      "progress_percentage": 85,
      "estimated_remaining_minutes": 10
    },
    "resource_usage": {
      "cpu_percent": 85.2,
      "memory_mb": 768,
      "disk_io_mb_per_sec": 12.3,
      "network_io_mb_per_sec": 4.5
    },
    "recommendations": [
      "Agent is near CPU capacity (85.2%). Consider offloading tasks.",
      "Agent has high success rate (100%). Can handle complex tasks.",
      "Agent produces high-quality code (0.92 score). Assign critical features."
    ]
  }
}
```

**Optimization Decisions:**
- ✅ Auth Specialist is high-performing but overloaded
- ✅ Offload integration testing to Database Architect (currently idle)
- ✅ Keep Auth Specialist on critical auth features only

---

### Phase 3: Enable Real-Time Swarm Monitoring

```javascript
// Step 3: Subscribe to real-time swarm activity updates
mcp__flow-nexus__swarm_monitor({
  duration: 300,    // Monitor for 5 minutes
  interval: 10      // Update every 10 seconds
})
```

**Expected Output (Streaming):**
```
🔍 Swarm Monitoring Started (5 minutes, 10-second intervals)

[10s] Swarm: swarm-ecommerce-2025
      Agents: 6 active, 4 busy, 2 idle
      Tasks: 5 running, 3 completed, 2 pending
      Avg CPU: 67.3% | Avg Memory: 512.8 MB

[20s] Swarm: swarm-ecommerce-2025
      Agents: 6 active, 4 busy, 2 idle
      Tasks: 5 running, 3 completed, 2 pending
      Avg CPU: 68.1% | Avg Memory: 518.2 MB
      ⚠️ Alert: agent-backend-auth-specialist-a1b2c3 CPU at 87.5%

[30s] Swarm: swarm-ecommerce-2025
      Agents: 6 active, 3 busy, 3 idle
      Tasks: 4 running, 4 completed, 2 pending
      Avg CPU: 52.3% | Avg Memory: 468.9 MB
      ✅ Completed: task-auth-service-001 (agent-backend-auth-specialist-a1b2c3)

[40s] Swarm: swarm-ecommerce-2025
      Agents: 6 active, 4 busy, 2 idle
      Tasks: 5 running, 4 completed, 1 pending
      Avg CPU: 61.7% | Avg Memory: 492.1 MB
      🚀 Started: task-api-gateway-004 (agent-api-designer-j1k2l3)
```

**Real-Time Insights:**
- ✅ Auth service completed after 90 minutes
- ⚠️ Auth Specialist CPU spiked to 87.5% during final tests
- ✅ API Designer now busy (API Gateway task started)
- ✅ Overall CPU dropped from 67% to 52% after task completion

---

### Phase 4: Scale Swarm for Increased Workload

**Scenario:** Security audit uncovered 12 OWASP violations requiring immediate fixes. Current agents are busy. Scale swarm to add 2 more coders.

```javascript
// Step 4: Scale swarm from 6 to 8 agents
mcp__flow-nexus__swarm_scale({
  target_agents: 8
})
```

**Expected Output:**
```json
{
  "swarm_id": "swarm-ecommerce-2025",
  "previous_agent_count": 6,
  "target_agent_count": 8,
  "status": "scaling",
  "scaling_progress": {
    "agents_to_spawn": 2,
    "agents_spawned": 0,
    "estimated_completion_seconds": 30
  },
  "message": "Scaling swarm from 6 to 8 agents. New agents will be spawned automatically."
}
```

**Auto-Spawning Process:**
```
🔄 Scaling Swarm: swarm-ecommerce-2025
   Current: 6 agents → Target: 8 agents

[5s]  Spawning agent 7/8: type=coder, capabilities=[security-fixes, owasp-compliance]
[10s] ✅ Spawned: agent-security-coder-s1t2u3
[15s] Spawning agent 8/8: type=analyst, capabilities=[code-review, vulnerability-testing]
[20s] ✅ Spawned: agent-security-tester-v4w5x6

[25s] ✅ Scaling Complete!
      Swarm now has 8 agents (6 original + 2 new)
      New agents: agent-security-coder-s1t2u3, agent-security-tester-v4w5x6
      Ready to accept OWASP fix tasks
```

**Benefits:**
- ✅ 2 new agents ready for OWASP violation fixes
- ✅ Original agents continue existing tasks uninterrupted
- ✅ Increased capacity for parallel security remediation

---

### Phase 5: Agent List After Scaling

```javascript
// Step 5: Verify new agents are active
mcp__flow-nexus__agent_list({
  filter: "active"
})
```

**Expected Output:**
```json
{
  "agents": [
    // Original 6 agents
    { "agent_id": "agent-backend-auth-specialist-a1b2c3", "status": "active" },
    { "agent_id": "agent-backend-catalog-specialist-d4e5f6", "status": "active" },
    { "agent_id": "agent-database-architect-g7h8i9", "status": "active" },
    { "agent_id": "agent-security-specialist-m4n5o6", "status": "active" },
    { "agent_id": "agent-devops-engineer-p7q8r9", "status": "active" },
    { "agent_id": "agent-api-designer-j1k2l3", "status": "active" },

    // New security-focused agents
    {
      "agent_id": "agent-security-coder-s1t2u3",
      "type": "coder",
      "name": "security-coder",
      "status": "active",
      "capabilities": ["security-fixes", "owasp-compliance", "input-sanitization"]
    },
    {
      "agent_id": "agent-security-tester-v4w5x6",
      "type": "analyst",
      "name": "security-tester",
      "status": "active",
      "capabilities": ["vulnerability-testing", "penetration-testing", "zap-scanning"]
    }
  ],
  "total": 8,
  "filter_applied": "active"
}
```

---

### Phase 6: Orchestrate OWASP Violation Fixes (Using New Agents)

```javascript
// Step 6: Assign OWASP fixes to new security-focused agents
mcp__flow-nexus__task_orchestrate({
  task: `Fix 12 OWASP Top 10 violations found in security audit:

  HIGH SEVERITY (Fix Immediately):
  1. SQL Injection in /catalog/search endpoint (use parameterized queries)
  2. Missing JWT validation in /orders/create endpoint (add auth middleware)
  3. Passwords stored in plain text in debug logs (remove logging)
  4. CORS allows all origins (restrict to production domain)

  MEDIUM SEVERITY:
  5. Missing rate limiting on /auth/login (10 attempts/minute)
  6. Error messages expose stack traces (sanitize errors)
  7. No HTTPS enforcement (redirect HTTP to HTTPS)
  8. Session tokens not rotated after login (implement rotation)

  LOW SEVERITY:
  9. Missing Content-Security-Policy header (add CSP)
  10. X-Frame-Options header missing (add DENY)
  11. npm audit shows 3 moderate vulnerabilities (update dependencies)
  12. Logging doesn't include security events (add audit logging)

  Deliverables:
  - /src/auth-service/middleware/jwt-validator.js (enhanced validation)
  - /src/catalog-service/routes/search.js (parameterized queries)
  - /src/api-gateway/middleware/cors.js (restricted origins)
  - /src/api-gateway/middleware/rate-limiter.js (enhanced limits)
  - /tests/security/owasp-fixes.test.js (regression tests)
  - /security/fixes-report.md (summary of changes)`,

  strategy: "parallel",        // Both agents work simultaneously
  priority: "critical",        // High severity vulnerabilities
  maxAgents: 2                 // Security Coder + Security Tester
})
```

**Expected Output:**
```json
{
  "task_id": "task-owasp-fixes-007",
  "status": "running",
  "strategy": "parallel",
  "priority": "critical",
  "assigned_agents": [
    "agent-security-coder-s1t2u3",      // Implements fixes
    "agent-security-tester-v4w5x6"     // Tests fixes
  ],
  "estimated_completion_time_minutes": 120,
  "started_at": "2025-11-02T13:00:00Z"
}
```

---

### Phase 7: Monitor Swarm Status After Scaling

```javascript
// Step 7: Check overall swarm health after scaling
mcp__flow-nexus__swarm_status({
  verbose: true
})
```

**Expected Output:**
```json
{
  "swarm_id": "swarm-ecommerce-2025",
  "topology": "hierarchical",
  "status": "healthy",
  "agents": {
    "total": 8,
    "active": 8,
    "idle": 1,
    "busy": 7
  },
  "coordinators": [
    {
      "id": "root-coordinator-001",
      "type": "root",
      "children": [
        "sub-coordinator-backend-services",
        "sub-coordinator-infrastructure",
        "sub-coordinator-quality-assurance",
        "sub-coordinator-security-team"        // New coordinator for security agents
      ]
    },
    {
      "id": "sub-coordinator-security-team",   // Auto-created during scaling
      "type": "domain",
      "agents": [
        "agent-security-coder-s1t2u3",
        "agent-security-tester-v4w5x6"
      ]
    }
  ],
  "tasks": {
    "total": 7,
    "completed": 4,
    "running": 3,
    "pending": 0,
    "failed": 0
  },
  "health_metrics": {
    "uptime_seconds": 3600,
    "total_tasks_completed": 4,
    "average_response_time_ms": 2500,
    "error_rate": 0.0,
    "average_cpu_percent": 72.5,
    "average_memory_mb": 548.3
  },
  "scaling_info": {
    "max_agents": 8,
    "current_agents": 8,
    "scaling_capacity_used_percent": 100,
    "can_scale_further": false,
    "recommendation": "Consider increasing maxAgents if more capacity needed"
  }
}
```

**Coordination Structure After Scaling:**
```
root-coordinator-001
├── sub-coordinator-backend-services
│   ├── backend-auth-specialist (coder) - IDLE
│   └── backend-catalog-specialist (coder) - BUSY
├── sub-coordinator-infrastructure
│   ├── database-architect (analyst) - BUSY
│   └── devops-engineer (optimizer) - BUSY
├── sub-coordinator-quality-assurance
│   ├── api-designer (researcher) - BUSY
│   └── security-specialist (analyst) - BUSY
└── sub-coordinator-security-team (NEW!)
    ├── security-coder (coder) - BUSY
    └── security-tester (analyst) - BUSY
```

**Health Check:**
- ✅ All 8 agents active and healthy
- ✅ 7 agents busy (87.5% utilization)
- ✅ 4 tasks completed, 3 running, 0 failed
- ✅ New sub-coordinator auto-created for security team
- ⚠️ Swarm at max capacity (100% of 8 agents)
- 💡 Recommendation: Increase `maxAgents` if more tasks queued

---

## Complete Code Example

```javascript
// ========================================
// FLOW NEXUS AGENT COORDINATION
// Real-Time Collaboration & Scaling
// ========================================

async function coordinateEcommerceSwarm() {
  console.log("🤝 Coordinating E-Commerce Swarm Agents...\n");

  // Phase 1: Monitor Performance
  console.log("📊 Phase 1: Analyzing Agent Performance");
  const metrics = await mcp__flow-nexus__agent_metrics({ metric: "all" });

  console.log(`   Total Agents: ${metrics.metrics.overall.total_agents}`);
  console.log(`   Busy Agents: ${metrics.metrics.overall.busy_agents}`);
  console.log(`   Avg CPU: ${metrics.metrics.overall.average_cpu_percent.toFixed(1)}%`);

  // Identify bottlenecks
  const overloaded = metrics.metrics.by_agent.filter(a => a.cpu_percent > 80);
  const idle = metrics.metrics.by_agent.filter(a => a.status === "idle");

  console.log(`   ⚠️ Overloaded: ${overloaded.map(a => a.name).join(", ")}`);
  console.log(`   💤 Idle: ${idle.map(a => a.name).join(", ")}`);

  // Phase 2: Deep Dive on Bottleneck
  if (overloaded.length > 0) {
    console.log(`\n🔍 Phase 2: Deep Dive on ${overloaded[0].name}`);
    const agentMetrics = await mcp__flow-nexus__agent_metrics({
      agentId: overloaded[0].agent_id,
      metric: "performance"
    });
    console.log(`   Success Rate: ${(agentMetrics.performance_metrics.success_rate * 100).toFixed(0)}%`);
    console.log(`   Avg Task Duration: ${agentMetrics.performance_metrics.average_task_duration_minutes} min`);
    console.log(`   Recommendations: ${agentMetrics.performance_metrics.recommendations.length}`);
  }

  // Phase 3: Real-Time Monitoring
  console.log("\n📡 Phase 3: Starting Real-Time Monitoring (60 seconds)");
  const monitorPromise = mcp__flow-nexus__swarm_monitor({
    duration: 60,
    interval: 10
  });

  // Phase 4: Scale Swarm (if needed)
  if (metrics.metrics.overall.busy_agents / metrics.metrics.overall.total_agents > 0.8) {
    console.log("\n🔄 Phase 4: Scaling Swarm (High Utilization Detected)");
    const scaling = await mcp__flow-nexus__swarm_scale({
      target_agents: metrics.metrics.overall.total_agents + 2
    });
    console.log(`   Scaling from ${scaling.previous_agent_count} to ${scaling.target_agent_count} agents`);

    // Wait for scaling to complete
    await new Promise(resolve => setTimeout(resolve, 30000));

    console.log("   ✅ Scaling Complete!");
  }

  // Phase 5: Verify Agent List
  console.log("\n📋 Phase 5: Verifying Active Agents");
  const agentList = await mcp__flow-nexus__agent_list({ filter: "active" });
  console.log(`   Active Agents: ${agentList.total}`);
  agentList.agents.forEach(agent => {
    console.log(`   - ${agent.name} (${agent.type})`);
  });

  // Phase 6: Orchestrate New Tasks with Scaled Swarm
  console.log("\n🎯 Phase 6: Orchestrating OWASP Fixes");
  const owaspTask = await mcp__flow-nexus__task_orchestrate({
    task: "Fix 12 OWASP Top 10 violations...",
    strategy: "parallel",
    priority: "critical",
    maxAgents: 2
  });
  console.log(`   Task ID: ${owaspTask.task_id}`);
  console.log(`   Status: ${owaspTask.status}`);

  // Wait for monitoring to complete
  await monitorPromise;

  // Phase 7: Final Swarm Status
  console.log("\n🏁 Phase 7: Final Swarm Status");
  const finalStatus = await mcp__flow-nexus__swarm_status({ verbose: true });
  console.log(`   Status: ${finalStatus.status}`);
  console.log(`   Agents: ${finalStatus.agents.total} (${finalStatus.agents.busy} busy)`);
  console.log(`   Tasks: ${finalStatus.tasks.completed} completed, ${finalStatus.tasks.running} running`);
  console.log(`   Avg CPU: ${finalStatus.health_metrics.average_cpu_percent.toFixed(1)}%`);
  console.log(`   Error Rate: ${(finalStatus.health_metrics.error_rate * 100).toFixed(1)}%`);

  return {
    agents: agentList.total,
    tasks_completed: finalStatus.tasks.completed,
    health: finalStatus.status
  };
}

// Execute coordination
coordinateEcommerceSwarm()
  .then(result => {
    console.log("\n🎉 Swarm Coordination Complete!");
    console.log(`   Agents: ${result.agents}`);
    console.log(`   Tasks Completed: ${result.tasks_completed}`);
    console.log(`   Health: ${result.health}`);
  })
  .catch(error => {
    console.error("❌ Coordination failed:", error);
  });
```

---

## Outcomes and Results

### Coordination Success Metrics

**Before Coordination:**
- 6 agents (1 overloaded, 2 idle)
- 3 tasks completed
- Auth Specialist at 85% CPU
- Database Architect idle at 12% CPU
- No security remediation capacity

**After Coordination:**
- 8 agents (all optimally utilized)
- 7 tasks completed
- Auth Specialist idle (task complete)
- 2 new security agents actively fixing OWASP violations
- Average CPU balanced at 72.5%

**Impact:**
- ✅ 33% increase in agent capacity (6 → 8)
- ✅ 133% increase in tasks completed (3 → 7)
- ✅ Eliminated bottleneck (Auth Specialist now idle)
- ✅ Security vulnerabilities being addressed in parallel
- ✅ 0% error rate maintained

---

## Pro Tips and Best Practices

### Tip 1: Monitor Before Scaling

**Always check metrics before scaling:**
```javascript
// ❌ DON'T: Scale blindly
await swarm_scale({ target_agents: 20 });  // Wasteful if agents idle

// ✅ DO: Check utilization first
const metrics = await agent_metrics({ metric: "all" });
const utilization = metrics.metrics.overall.busy_agents / metrics.metrics.overall.total_agents;

if (utilization > 0.8) {
  await swarm_scale({ target_agents: currentAgents + 2 });
}
```

### Tip 2: Use Interval Monitoring for Long Tasks

```javascript
// For tasks > 30 minutes, monitor every 30-60 seconds
await swarm_monitor({
  duration: 3600,    // 1 hour
  interval: 60       // Update every minute
});

// For quick tasks, monitor every 5-10 seconds
await swarm_monitor({
  duration: 300,     // 5 minutes
  interval: 10       // Update every 10 seconds
});
```

### Tip 3: Scale Down When Tasks Complete

```javascript
// After peak workload, scale down to save resources
const status = await swarm_status({ verbose: true });

if (status.agents.idle > status.agents.total * 0.5) {
  console.log("⬇️ Scaling down (50%+ agents idle)");
  await swarm_scale({
    target_agents: Math.ceil(status.agents.total * 0.6)
  });
}
```

### Tip 4: Agent Specialization Over Generalization

```javascript
// ✅ DO: Spawn specialized agents
await agent_spawn({
  type: "coder",
  capabilities: ["security-fixes", "owasp-compliance"]
});

// ❌ DON'T: Spawn generic agents
await agent_spawn({
  type: "coder",
  capabilities: ["coding"]  // Too vague
});
```

### Tip 5: Track Agent Performance Over Time

```javascript
// Store metrics in database for trending
const metrics = await agent_metrics({ agentId: "agent-123" });

await database.insert("agent_metrics", {
  agent_id: metrics.agent_id,
  timestamp: new Date(),
  cpu_percent: metrics.resource_usage.cpu_percent,
  tasks_completed: metrics.performance_metrics.tasks_completed,
  success_rate: metrics.performance_metrics.success_rate
});

// Query for trends
const trend = await database.query(`
  SELECT
    DATE(timestamp) as date,
    AVG(cpu_percent) as avg_cpu,
    SUM(tasks_completed) as total_tasks
  FROM agent_metrics
  WHERE agent_id = 'agent-123'
    AND timestamp > NOW() - INTERVAL '7 days'
  GROUP BY DATE(timestamp)
`);
```

---

## Common Pitfalls to Avoid

### ❌ Pitfall 1: Over-Scaling Too Quickly
```javascript
// DON'T: Double agents immediately
await swarm_scale({ target_agents: currentAgents * 2 });
// Result: Resource waste, high costs

// DO: Incremental scaling
await swarm_scale({ target_agents: currentAgents + 2 });
// Wait 5 minutes, reassess
```

### ❌ Pitfall 2: Ignoring Idle Agents
```javascript
// DON'T: Scale up while agents idle
if (pendingTasks > 0) {
  await swarm_scale({ target_agents: currentAgents + 5 });
}
// Result: Paying for unused agents

// DO: Assign tasks to idle agents first
const idleAgents = await agent_list({ filter: "idle" });
if (idleAgents.total > 0) {
  // Assign tasks to idle agents
} else {
  await swarm_scale({ target_agents: currentAgents + 2 });
}
```

### ❌ Pitfall 3: Not Monitoring After Scaling
```javascript
// DON'T: Scale and forget
await swarm_scale({ target_agents: 10 });
// Never check if new agents are utilized

// DO: Monitor post-scaling
await swarm_scale({ target_agents: 10 });
await sleep(60000);  // Wait 1 minute
const metrics = await agent_metrics({ metric: "all" });
console.log(`Utilization: ${metrics.metrics.overall.busy_agents / 10 * 100}%`);
```

---

## Next Steps

After agent coordination:

1. **Optimize Workflows** → Reduce task duration with better resource allocation
2. **Implement Auto-Scaling** → Automatically scale based on utilization thresholds
3. **Set Up Alerts** → Notify when agents fail or CPU exceeds 90%
4. **Analyze Performance Trends** → Identify patterns in agent efficiency
5. **Deploy to Production** → Use DevOps agent for containerized deployment

---

## Related Examples

- **Example 1**: Swarm Initialization (Creating the foundation)
- **Example 2**: Task Orchestration (Assigning work to agents)

---

**Flow Nexus Version**: 1.5.0
**Last Updated**: 2025-11-02
**Skill**: `flow-nexus-swarm`
