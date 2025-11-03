# Example 3: Adaptive Scaling for Dynamic Agent Management

## Scenario Description

You're building a SaaS platform with **unpredictable workload spikes**. During normal hours, 3-4 agents handle development tasks. But during critical events (customer demos, production incidents, major releases), you need to **scale up to 15+ agents** rapidly, then **scale back down** when the spike ends.

**Project Components**:
- Real-time monitoring dashboard (React + WebSockets)
- Multi-tenant API backend (Node.js + PostgreSQL)
- Authentication service (OAuth 2.0 + JWT)
- Payment processing (Stripe + webhook handlers)
- Email notification service (SendGrid)
- Caching layer (Redis)
- CI/CD pipeline (GitHub Actions + Docker)

**Why Adaptive Scaling?**
- **Cost efficiency**: Only use resources when needed
- **Performance**: Scale up for critical tasks, avoid bottlenecks
- **Flexibility**: Dynamically adjust to changing priorities
- **Auto-healing**: Replace failed agents automatically

**Use Cases**:
1. **Production Incident** (3 agents → 12 agents in 5 minutes)
2. **Feature Sprint** (4 agents → 8 agents for 2 weeks)
3. **Performance Optimization** (5 agents → 3 agents after bottleneck resolved)

---

## Step-by-Step Walkthrough

### Step 1: Initialize Swarm with Adaptive Scaling

```bash
# Initialize with adaptive strategy (auto-scales based on workload)
npx claude-flow@alpha swarm init hierarchical \
  --max-agents 15 \
  --strategy adaptive
```

**Expected Output**:
```json
{
  "swarmId": "swarm-saas-789",
  "topology": "hierarchical",
  "maxAgents": 15,
  "strategy": "adaptive",
  "currentAgents": 0,
  "scalingPolicy": {
    "minAgents": 2,
    "maxAgents": 15,
    "scaleUpThreshold": "high_priority_tasks > 5",
    "scaleDownThreshold": "idle_time > 30min",
    "cooldownPeriod": "5min"
  },
  "status": "initialized"
}
```

**What Happens**:
- Sets adaptive scaling strategy
- Defines min (2) and max (15) agent limits
- Configures auto-scaling triggers
- Enables dynamic agent spawning/termination

---

### Step 2: Start with Baseline Agents (Normal Load)

```javascript
// During normal development (no incidents, no sprints)
// Start with 3 baseline agents

[Single Message - Baseline Agent Spawning]:

Task("Backend Developer",
  "Maintain API endpoints, fix minor bugs, review PRs. Monitor for production incidents. Scale up if critical issues detected.",
  "coder")

Task("Frontend Developer",
  "Update React components, improve UX, optimize performance. Handle dashboard updates. Scale up for major feature work.",
  "coder")

Task("DevOps Engineer",
  "Monitor infrastructure, maintain CI/CD, optimize costs. Auto-spawn additional agents if deployment issues or scaling events occur.",
  "cicd-engineer")

// Store baseline configuration
mcp__memory__store({
  key: "swarm/config/baseline-agents",
  value: JSON.stringify({
    agents: ["Backend Developer", "Frontend Developer", "DevOps Engineer"],
    count: 3,
    purpose: "Normal development operations",
    scaleTriggers: ["production_incident", "feature_sprint", "performance_issue"]
  }),
  metadata: {
    tags: ["WHO:devops-engineer", "WHEN:" + Date.now(), "PROJECT:saas-adaptive", "WHY:configuration"],
    retention: "long-term"
  }
})
```

---

### Step 3: Trigger Scale-Up Event (Production Incident)

**Scenario**: Payment processing API returns 500 errors. Customer transactions failing.

**Incident Detection**:
```javascript
// DevOps Engineer detects incident via monitoring
const incidentDetected = {
  type: "production_incident",
  severity: "critical",
  service: "payment-api",
  errorRate: "35%",
  affectedCustomers: 47,
  detectedAt: new Date().toISOString()
};

// Store incident in memory
await mcp__memory__store({
  key: "swarm/incidents/payment-api-001",
  value: JSON.stringify(incidentDetected),
  metadata: {
    tags: ["WHO:devops-engineer", "WHEN:" + Date.now(), "PROJECT:saas-adaptive", "WHY:incident"],
    retention: "long-term"
  }
});

// Trigger adaptive scaling
await hooks.notify({
  message: "CRITICAL INCIDENT: Payment API 35% error rate. Triggering adaptive scale-up to 12 agents."
});
```

**Auto-Scale Up to Incident Response Team**:
```javascript
// Adaptive coordinator automatically spawns incident response team
[Single Message - Incident Response Scale-Up]:

// Incident Commander (coordinator)
Task("Incident Commander",
  "Coordinate incident response. Assign tasks to specialists. Monitor progress. Communicate with stakeholders every 15 minutes.",
  "coordinator")

// Database Specialists (2 agents)
Task("Database Investigator",
  "Check PostgreSQL logs for payment table locks, slow queries, connection pool exhaustion. Report findings to Incident Commander.",
  "analyst")

Task("Database Performance Optimizer",
  "Optimize slow payment queries. Add indexes if needed. Monitor query execution plans. Coordinate with Database Investigator.",
  "optimizer")

// Backend Specialists (3 agents)
Task("Payment API Debugger",
  "Debug payment processing logic. Check Stripe webhook handlers. Review error logs for root cause. Report to Incident Commander.",
  "coder")

Task("API Performance Analyst",
  "Profile API response times. Identify bottlenecks in payment flow. Check rate limits and timeouts.",
  "analyst")

Task("Backend Hotfix Developer",
  "Implement urgent fixes for payment API based on findings. Deploy to staging first. Coordinate with DevOps for production deployment.",
  "coder")

// Infrastructure Specialists (2 agents)
Task("Infrastructure Scaler",
  "Scale up API instances if needed. Check CPU/memory usage. Monitor Redis cache hit rates. Add capacity if infrastructure-limited.",
  "cicd-engineer")

Task("Network Debugger",
  "Check network latency to Stripe API. Verify webhook delivery. Inspect load balancer health. Report infrastructure issues.",
  "analyst")

// Testing & Validation (2 agents)
Task("Payment Flow Tester",
  "Test payment flows in staging. Validate fixes don't break existing functionality. Run regression suite.",
  "tester")

Task("Customer Impact Analyst",
  "Identify affected customers. Prepare communication draft. Estimate financial impact. Track resolution progress.",
  "analyst")

// Total: 12 agents (3 baseline + 9 incident response)
```

**Scaling Metrics**:
```javascript
// Track scaling event
mcp__memory__store({
  key: "swarm/scaling/incident-payment-api-001",
  value: JSON.stringify({
    event: "scale_up",
    trigger: "production_incident",
    fromAgents: 3,
    toAgents: 12,
    duration: "4min 32sec",
    reason: "Payment API 35% error rate affecting 47 customers",
    spawnedAgents: [
      "Incident Commander",
      "Database Investigator",
      "Database Performance Optimizer",
      "Payment API Debugger",
      "API Performance Analyst",
      "Backend Hotfix Developer",
      "Infrastructure Scaler",
      "Network Debugger",
      "Payment Flow Tester",
      "Customer Impact Analyst"
    ],
    expectedResolutionTime: "2 hours",
    timestamp: new Date().toISOString()
  }),
  metadata: {
    tags: ["WHO:incident-commander", "WHEN:" + Date.now(), "PROJECT:saas-adaptive", "WHY:scaling"],
    retention: "long-term"
  }
})
```

---

### Step 4: Incident Resolution & Intelligent Scale-Down

**Root Cause Found**:
```javascript
// Database Investigator finds root cause
const rootCause = {
  issue: "Payment table lock contention during high-volume transactions",
  cause: "Missing index on payment_transactions.created_at",
  fix: "ALTER TABLE payment_transactions ADD INDEX idx_created_at (created_at)",
  impact: "35% error rate reduced to 0.2% after index added",
  resolvedBy: "Database Performance Optimizer",
  resolvedAt: new Date().toISOString(),
  timeToResolve: "1 hour 47 minutes"
};

await mcp__memory__store({
  key: "swarm/resolutions/payment-api-001",
  value: JSON.stringify(rootCause),
  metadata: {
    tags: ["WHO:database-performance-optimizer", "WHEN:" + Date.now(), "PROJECT:saas-adaptive", "WHY:resolution"],
    retention: "long-term"
  }
});
```

**Gradual Scale-Down**:
```javascript
// Incident Commander initiates controlled scale-down
// Phase 1: Terminate incident-specific agents (within 15 minutes)
const agentsToTerminate = [
  "Database Investigator",      // Investigation complete
  "API Performance Analyst",    // Bottleneck identified
  "Network Debugger",           // No network issues found
  "Customer Impact Analyst"     // Impact assessment complete
];

await hooks.notify({
  message: "Incident resolved. Initiating Phase 1 scale-down: terminating 4 investigation agents."
});

// Phase 2: Keep specialist agents for validation (30 minutes)
// Retain: Database Performance Optimizer, Backend Hotfix Developer, Payment Flow Tester

// Phase 3: Final scale-down to baseline + on-call (after 2 hours)
const retainedAgents = [
  "Backend Developer",          // Baseline
  "Frontend Developer",         // Baseline
  "DevOps Engineer",           // Baseline
  "Backend Hotfix Developer"   // On-call for 24 hours (monitor for regressions)
];

await mcp__memory__store({
  key: "swarm/scaling/incident-payment-api-001-scaledown",
  value: JSON.stringify({
    event: "scale_down",
    trigger: "incident_resolved",
    fromAgents: 12,
    toAgents: 4,
    phases: [
      { phase: 1, duration: "15min", terminated: 4, retained: 8 },
      { phase: 2, duration: "30min", terminated: 4, retained: 4 },
      { phase: 3, duration: "2hr", terminated: 0, retained: 4 }
    ],
    onCallAgent: "Backend Hotfix Developer",
    onCallDuration: "24 hours",
    timestamp: new Date().toISOString()
  }),
  metadata: {
    tags: ["WHO:incident-commander", "WHEN:" + Date.now(), "PROJECT:saas-adaptive", "WHY:scaling"],
    retention: "long-term"
  }
})
```

---

### Step 5: Auto-Scaling Based on Task Queue Depth

**Scenario**: Feature sprint starts, 15 high-priority tasks added to backlog

**Queue Monitoring**:
```javascript
// DevOps Engineer monitors task queue
const taskQueue = await mcp__memory__vector_search({
  query: "high priority tasks in backlog",
  mode: "execution",
  topK: 50
});

const highPriorityTasks = taskQueue.results.filter(r =>
  r.content.includes("priority: high") && r.content.includes("status: pending")
);

if (highPriorityTasks.length > 5) {
  // Auto-trigger scale-up
  const agentsNeeded = Math.min(
    Math.ceil(highPriorityTasks.length / 2), // 2 tasks per agent
    15 - 3 // Don't exceed max (15) minus baseline (3)
  );

  await hooks.notify({
    message: `Auto-scaling: ${highPriorityTasks.length} high-priority tasks detected. Spawning ${agentsNeeded} additional agents.`
  });

  // Spawn feature development team
  for (let i = 0; i < agentsNeeded; i++) {
    await Task(`Feature Developer ${i + 1}`,
      `Pick high-priority task from backlog. Implement, test, and deploy. Coordinate with baseline team.`,
      "coder"
    );
  }
}
```

**Auto-Scale Down After Completion**:
```javascript
// Monitor idle agents
setInterval(async () => {
  const agentMetrics = await mcp__claude-flow__agent_metrics({ metric: "tasks" });

  // Find agents idle for 30+ minutes
  const idleAgents = agentMetrics.agents.filter(agent =>
    agent.idleTime > 30 * 60 * 1000 && // 30 minutes in ms
    !agent.name.includes("Baseline") // Don't terminate baseline agents
  );

  if (idleAgents.length > 0) {
    await hooks.notify({
      message: `Auto-scaling: ${idleAgents.length} agents idle for 30+ minutes. Terminating to reduce costs.`
    });

    // Terminate idle agents
    for (const agent of idleAgents) {
      await terminateAgent(agent.id);
    }
  }
}, 10 * 60 * 1000); // Check every 10 minutes
```

---

### Step 6: Monitor Scaling Metrics

```bash
# View current swarm size
npx claude-flow@alpha swarm status

# Check scaling history
npx claude-flow@alpha memory retrieve --key "swarm/scaling/*"

# Get agent utilization metrics
npx claude-flow@alpha agent metrics --metric tasks

# View cost savings from auto-scaling
npx claude-flow@alpha hooks session-end --export-metrics true
```

**Sample Metrics Output**:
```json
{
  "scalingEvents": [
    { "event": "scale_up", "from": 3, "to": 12, "reason": "production_incident", "duration": "4min 32sec" },
    { "event": "scale_down", "from": 12, "to": 4, "reason": "incident_resolved", "duration": "2hr 15min" },
    { "event": "scale_up", "from": 4, "to": 10, "reason": "high_priority_tasks", "duration": "3min 18sec" },
    { "event": "scale_down", "from": 10, "to": 3, "reason": "tasks_complete", "duration": "1hr 45min" }
  ],
  "costSavings": {
    "agentHoursAvoided": 87,
    "estimatedCostSavings": "$347 USD (compared to always running 15 agents)",
    "averageTeamSize": 5.3,
    "peakTeamSize": 12,
    "minTeamSize": 3
  },
  "performance": {
    "incidentResolutionTime": "1hr 47min (40% faster than historical average)",
    "featureSprintThroughput": "23 tasks completed in 3 days (2.1x baseline)",
    "agentUtilization": "76% average (vs 42% with static team)"
  }
}
```

---

## Expected Outcomes

### Successful Adaptive Scaling Results:

1. **Cost Efficiency**
   - 87 agent-hours saved per month
   - $347 monthly savings (vs static 15-agent team)
   - Average team size: 5.3 agents (vs static 15)

2. **Performance Gains**
   - Incident resolution: 40% faster (scale-up when needed)
   - Feature throughput: 2.1x improvement during sprints
   - Agent utilization: 76% (vs 42% static)

3. **Flexibility**
   - Scale from 3 → 12 agents in <5 minutes
   - Auto-scale down after 30 minutes idle
   - Dynamic response to changing priorities

4. **Auto-Healing**
   - Replace failed agents automatically
   - Redistribute tasks to healthy agents
   - No manual intervention required

---

## Code Examples

### Example: Auto-Scale Up Based on Error Rate

```javascript
// Monitor production error rate and auto-scale for incidents
async function monitorErrorRateAndScale() {
  // Query monitoring system for error rate
  const errorRate = await getErrorRateFromMonitoring("payment-api");

  if (errorRate > 5) { // 5% threshold
    const currentAgents = await mcp__claude-flow__swarm_status();

    // Calculate agents needed based on severity
    const severity = errorRate > 20 ? "critical" : "high";
    const targetAgents = severity === "critical" ? 12 : 8;

    if (currentAgents.activeAgents < targetAgents) {
      // Auto-scale up
      await hooks.notify({
        message: `ERROR RATE SPIKE: ${errorRate}% on payment-api. Auto-scaling from ${currentAgents.activeAgents} to ${targetAgents} agents.`
      });

      // Spawn incident response team
      const agentsToSpawn = targetAgents - currentAgents.activeAgents;
      for (let i = 0; i < agentsToSpawn; i++) {
        await Task(`Incident Responder ${i + 1}`,
          "Investigate payment API errors. Check logs, database, network. Report findings.",
          "analyst"
        );
      }

      // Track scaling event
      await mcp__memory__store({
        key: `swarm/scaling/auto-${Date.now()}`,
        value: JSON.stringify({
          trigger: "error_rate_spike",
          errorRate: `${errorRate}%`,
          from: currentAgents.activeAgents,
          to: targetAgents,
          timestamp: new Date().toISOString()
        }),
        metadata: {
          tags: ["WHO:devops-engineer", "WHEN:" + Date.now(), "PROJECT:saas-adaptive", "WHY:auto-scaling"],
          retention: "mid-term"
        }
      });
    }
  }
}

// Run every 5 minutes
setInterval(monitorErrorRateAndScale, 5 * 60 * 1000);
```

### Example: Intelligent Scale-Down with Cooldown Period

```javascript
// Scale down agents after cooldown period to prevent flapping
async function intelligentScaleDown() {
  const agents = await mcp__claude-flow__agent_list({ filter: "idle" });

  // Find agents idle for 30+ minutes
  const now = Date.now();
  const cooldownPeriod = 30 * 60 * 1000; // 30 minutes

  for (const agent of agents) {
    const idleDuration = now - agent.lastTaskCompletedAt;

    if (idleDuration > cooldownPeriod && !agent.isBaseline) {
      // Check if in scale-down cooldown (prevent rapid scale down/up)
      const lastScaleDown = await mcp__memory__vector_search({
        query: "last scale down event timestamp",
        mode: "execution",
        topK: 1
      });

      const timeSinceLastScaleDown = lastScaleDown.results[0]
        ? now - parseInt(lastScaleDown.results[0].content.timestamp)
        : Infinity;

      if (timeSinceLastScaleDown > 5 * 60 * 1000) { // 5 minute cooldown
        await hooks.notify({
          message: `Auto-scaling: Terminating ${agent.name} (idle for ${Math.floor(idleDuration / 60000)} minutes)`
        });

        await terminateAgent(agent.id);

        // Update last scale-down timestamp
        await mcp__memory__store({
          key: "swarm/scaling/last-scaledown",
          value: JSON.stringify({ timestamp: now }),
          metadata: {
            tags: ["WHO:system", "WHEN:" + Date.now(), "PROJECT:saas-adaptive", "WHY:auto-scaling"],
            retention: "short-term"
          }
        });
      }
    }
  }
}

setInterval(intelligentScaleDown, 10 * 60 * 1000); // Check every 10 minutes
```

### Example: Predictive Scaling Based on Historical Patterns

```javascript
// Use historical data to predict scaling needs
async function predictiveScaling() {
  // Query historical scaling events
  const history = await mcp__memory__vector_search({
    query: "scaling events in last 30 days",
    mode: "planning",
    topK: 100
  });

  // Analyze patterns (e.g., feature sprints every Monday, incidents on Fridays)
  const dayOfWeek = new Date().getDay();
  const hourOfDay = new Date().getHours();

  // Monday mornings: Feature sprint kickoff
  if (dayOfWeek === 1 && hourOfDay >= 9 && hourOfDay <= 10) {
    const avgMondayAgents = calculateAverageAgents(history, "Monday 9-10am");

    if (avgMondayAgents > 5) {
      await hooks.notify({
        message: `Predictive scaling: Monday 9am sprint kickoff detected. Pre-scaling to ${avgMondayAgents} agents.`
      });

      await scaleToTargetAgents(avgMondayAgents);
    }
  }

  // Friday afternoons: Historically high incident rate
  if (dayOfWeek === 5 && hourOfDay >= 14) {
    await hooks.notify({
      message: "Predictive scaling: Friday afternoon. Keeping 2 extra agents on-call for potential incidents."
    });

    await ensureMinimumAgents(5); // 3 baseline + 2 on-call
  }
}

setInterval(predictiveScaling, 60 * 60 * 1000); // Check hourly
```

---

## Tips and Best Practices

### 1. Define Clear Baseline
**Always maintain minimum viable team**:
- 2-4 baseline agents for core operations
- Don't scale below baseline even during idle periods
- Baseline agents handle routine tasks + monitoring

### 2. Cooldown Periods Prevent Flapping
**Avoid rapid scale-up/down cycles**:
- Wait 5 minutes between scale-down events
- Require 30 minutes idle before termination
- Use exponential backoff for repeated scaling

### 3. Gradual Scale-Down
**Don't terminate all agents at once**:
```javascript
// ✅ GOOD: Phased scale-down
Phase 1: Terminate investigation agents (immediate)
Phase 2: Terminate specialist agents (after 30 min)
Phase 3: Return to baseline (after 2 hours)

// ❌ BAD: Instant scale-down
Terminate all 9 agents immediately after incident resolved
```

### 4. Cost Tracking
**Monitor scaling costs**:
```bash
# Export cost metrics
npx claude-flow@alpha hooks session-end --export-metrics true

# Track agent-hours per project
npx claude-flow@alpha memory retrieve --key "swarm/scaling/*" | jq '.[] | .agentHours'
```

### 5. Auto-Healing with Replacement
**Replace failed agents automatically**:
```javascript
// Detect agent failure
if (agent.status === "failed") {
  await hooks.notify({
    message: `Agent ${agent.name} failed. Spawning replacement.`
  });

  // Spawn identical replacement
  await Task(agent.name, agent.instructions, agent.type);
}
```

### 6. Predictive Scaling
**Use historical data**:
- Monday mornings: Feature sprints (scale up)
- Friday afternoons: Incident-prone (keep on-call agents)
- Month-end: Payment processing spikes (pre-scale)

### 7. Memory MCP for Scaling Decisions
**Store scaling events for analysis**:
```javascript
mcp__memory__store({
  key: "swarm/scaling/event-123",
  value: JSON.stringify({
    trigger: "high_priority_tasks",
    from: 3,
    to: 8,
    duration: "3hr 45min",
    tasksCompleted: 12,
    costPerTask: "$0.47"
  }),
  metadata: {
    tags: ["WHO:system", "WHEN:" + Date.now(), "PROJECT:saas-adaptive", "WHY:analytics"],
    retention: "long-term"
  }
})
```

### 8. Set Max/Min Limits
**Prevent runaway scaling**:
- Min agents: 2 (always-on baseline)
- Max agents: 15 (cost cap)
- Per-event max: 12 (prevent over-scaling for single incident)

---

## Common Pitfalls to Avoid

1. **Scaling Too Slowly**: 5-minute scale-up is acceptable, 20 minutes is not
2. **No Cooldown**: Flapping (scale up/down/up) wastes resources
3. **Ignoring Costs**: Track agent-hours and set budgets
4. **Static Thresholds**: Adapt thresholds based on historical data
5. **No Baseline**: Terminating all agents during idle periods
6. **Instant Scale-Down**: Gradual phase-out prevents regressions
7. **No Metrics**: Track scaling events for optimization
8. **Manual Intervention**: Automate scale-up/down triggers

---

## When to Use Adaptive Scaling

### Use Adaptive Scaling When:
- ✅ Workload is unpredictable (incidents, spikes, sprints)
- ✅ Cost efficiency is a priority
- ✅ Tasks have varying urgency levels
- ✅ Team size fluctuates (3-15 agents)
- ✅ Performance SLAs require rapid response

### Use Static Sizing When:
- ✅ Workload is predictable and consistent
- ✅ Team size is fixed (e.g., 5 agents always)
- ✅ Cost is not a constraint
- ✅ Scaling overhead outweighs benefits

---

## Next Steps

After mastering adaptive scaling:
1. Combine with **hierarchical topology** for large-scale projects
2. Integrate **predictive ML models** for smarter scaling decisions
3. Implement **cost budgets** and auto-alerts for overspending
4. Use **A/B testing** to optimize scaling thresholds
5. Apply to **CI/CD pipelines** for dynamic test parallelization
