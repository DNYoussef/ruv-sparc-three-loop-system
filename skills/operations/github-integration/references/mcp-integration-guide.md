# MCP Tool Integration Guide for GitHub Skills

This guide documents how to integrate Model Context Protocol (MCP) tools with GitHub integration skills for enhanced coordination and automation.

## Overview

The GitHub integration skills leverage two primary MCP servers:

1. **Claude Flow** - Core swarm coordination and agent management
2. **Flow-Nexus** (optional) - Advanced cloud-based features and GitHub integration

## Setup Instructions

### Required: Claude Flow

Install Claude Flow MCP server (required for all GitHub skills):

```bash
# Add Claude Flow MCP server
claude mcp add claude-flow npx claude-flow@alpha mcp start
```

Verify installation:

```bash
# Check MCP servers
claude mcp list

# Test Claude Flow
npx claude-flow@alpha --version
```

### Optional: Flow-Nexus

Install Flow-Nexus for advanced features:

```bash
# Add Flow-Nexus MCP server
claude mcp add flow-nexus npx flow-nexus@latest mcp start
```

Register and authenticate:

```bash
# Register account
npx flow-nexus@latest register

# Login
npx flow-nexus@latest login
```

## Core MCP Tools

### Claude Flow Tools

#### Swarm Initialization

```javascript
// Initialize swarm with specific topology
mcp__claude-flow__swarm_init({
  topology: "mesh" | "hierarchical" | "ring" | "star",
  maxAgents: 5,
  strategy: "balanced" | "specialized" | "adaptive"
})
```

**Topology Selection Guide:**

- **Mesh**: PR code review - enables peer-to-peer agent communication
- **Hierarchical**: Multi-repo management, release management - centralized coordination
- **Star**: Project management - central coordinator with specialized agents
- **Ring**: Sequential workflows with circular dependencies

#### Agent Spawning

```javascript
// Spawn specialized agent
mcp__claude-flow__agent_spawn({
  type: "researcher" | "coder" | "analyst" | "optimizer" | "coordinator",
  name: "security-auditor",  // Custom name
  capabilities: ["security-scanning", "vulnerability-detection"]
})
```

**Agent Type Mapping:**

- `researcher`: Issue tracker, tester, docs writer
- `coder`: CI/CD engineer, workflow automation
- `analyst`: Code analyzer, security auditor, reviewer
- `optimizer`: Performance analyzer
- `coordinator`: Release manager, hierarchical coordinator, project coordinator

#### Task Orchestration

```javascript
// Orchestrate complex task across swarm
mcp__claude-flow__task_orchestrate({
  task: "Review PR #123 for security and performance",
  strategy: "parallel" | "sequential" | "adaptive",
  maxAgents: 4,
  priority: "low" | "medium" | "high" | "critical"
})
```

#### Monitoring and Status

```javascript
// Get swarm status
mcp__claude-flow__swarm_status({ verbose: true })

// List active agents
mcp__claude-flow__agent_list({ filter: "active" | "idle" | "busy" })

// Get agent performance metrics
mcp__claude-flow__agent_metrics({
  agentId: "agent-123",  // Optional: specific agent
  metric: "all" | "cpu" | "memory" | "tasks" | "performance"
})

// Check task status
mcp__claude-flow__task_status({
  taskId: "task-456",  // Optional: specific task
  detailed: true
})

// Retrieve task results
mcp__claude-flow__task_results({
  taskId: "task-456",
  format: "summary" | "detailed" | "raw"
})
```

### Flow-Nexus GitHub Tools (Optional)

#### Repository Analysis

```javascript
// Analyze GitHub repository
mcp__flow-nexus__github_repo_analyze({
  repo: "owner/repository",
  analysis_type: "code_quality" | "performance" | "security"
})
```

#### Workflow Monitoring

```javascript
// Subscribe to real-time workflow updates
mcp__flow-nexus__execution_stream_subscribe({
  stream_type: "claude-code" | "claude-flow-swarm" | "github-integration",
  sandbox_id: "sandbox-123"
})

// Get execution stream status
mcp__flow-nexus__execution_stream_status({
  stream_id: "stream-456"
})

// List files created during execution
mcp__flow-nexus__execution_files_list({
  stream_id: "stream-456",
  created_by: "claude-code" | "claude-flow" | "git-clone"
})
```

#### Real-time Subscriptions

```javascript
// Subscribe to GitHub events
mcp__flow-nexus__realtime_subscribe({
  table: "issues" | "pull_requests" | "releases",
  event: "INSERT" | "UPDATE" | "DELETE" | "*",
  filter: "status='open'"  // Optional SQL filter
})

// Unsubscribe from events
mcp__flow-nexus__realtime_unsubscribe({
  subscription_id: "sub-789"
})

// List active subscriptions
mcp__flow-nexus__realtime_list()
```

## Integration Patterns by Skill

### 1. Code Review Skill

**MCP Tool Usage:**

```javascript
// Step 1: Initialize mesh swarm for parallel review
mcp__claude-flow__swarm_init({
  topology: "mesh",
  maxAgents: 5,
  strategy: "specialized"
})

// Step 2: Spawn specialized reviewers
mcp__claude-flow__agent_spawn({ type: "analyst", name: "security-auditor" })
mcp__claude-flow__agent_spawn({ type: "optimizer", name: "perf-analyzer" })
mcp__claude-flow__agent_spawn({ type: "analyst", name: "code-analyzer" })
mcp__claude-flow__agent_spawn({ type: "researcher", name: "tester" })
mcp__claude-flow__agent_spawn({ type: "researcher", name: "reviewer" })

// Step 3 (Optional): Use Flow-Nexus for advanced GitHub analysis
mcp__flow-nexus__github_repo_analyze({
  repo: "owner/repo",
  analysis_type: "code_quality"
})

// Step 4: Monitor review progress
mcp__claude-flow__swarm_status({ verbose: true })
```

**Coordination Flow:**

1. MCP tools set up coordination topology
2. Claude Code's Task tool spawns actual agents to do work
3. Agents coordinate via shared memory (hooks)
4. MCP tools monitor progress and aggregate results

### 2. Multi-Repository Skill

**MCP Tool Usage:**

```javascript
// Step 1: Initialize hierarchical swarm for centralized control
mcp__claude-flow__swarm_init({
  topology: "hierarchical",
  maxAgents: 8,
  strategy: "adaptive"
})

// Step 2: Spawn coordinator and specialists
mcp__claude-flow__agent_spawn({ type: "coordinator", name: "hierarchical-coordinator" })
mcp__claude-flow__agent_spawn({ type: "analyst", name: "repo-architect" })
mcp__claude-flow__agent_spawn({ type: "analyst", name: "code-analyzer" })

// Step 3: Orchestrate multi-repo task
mcp__claude-flow__task_orchestrate({
  task: "Propagate API changes across 10 repositories",
  strategy: "sequential",  // Respect dependency order
  maxAgents: 10,
  priority: "high"
})

// Step 4: Monitor task execution
mcp__claude-flow__task_status({ detailed: true })
```

### 3. Project Management Skill

**MCP Tool Usage:**

```javascript
// Step 1: Initialize star swarm for centralized project management
mcp__claude-flow__swarm_init({
  topology: "star",
  maxAgents: 6,
  strategy: "balanced"
})

// Step 2: Spawn project management agents
mcp__claude-flow__agent_spawn({ type: "coordinator", name: "coordinator" })
mcp__claude-flow__agent_spawn({ type: "researcher", name: "issue-tracker" })
mcp__claude-flow__agent_spawn({ type: "researcher", name: "planner" })
mcp__claude-flow__agent_spawn({ type: "coordinator", name: "project-board-sync" })

// Step 3 (Optional): Subscribe to real-time issue updates
mcp__flow-nexus__realtime_subscribe({
  table: "issues",
  event: "INSERT",
  filter: "repository='owner/repo'"
})

// Step 4: Monitor agent activity
mcp__claude-flow__agent_metrics({ metric: "tasks" })
```

### 4. Release Management Skill

**MCP Tool Usage:**

```javascript
// Step 1: Initialize hierarchical swarm for release coordination
mcp__claude-flow__swarm_init({
  topology: "hierarchical",
  maxAgents: 8,
  strategy: "specialized"
})

// Step 2: Spawn release team
mcp__claude-flow__agent_spawn({ type: "coordinator", name: "release-manager" })
mcp__claude-flow__agent_spawn({ type: "coder", name: "cicd-engineer" })
mcp__claude-flow__agent_spawn({ type: "researcher", name: "tester" })
mcp__claude-flow__agent_spawn({ type: "analyst", name: "reviewer" })

// Step 3: Orchestrate release workflow
mcp__claude-flow__task_orchestrate({
  task: "Execute release v2.0.0 with validation",
  strategy: "sequential",
  maxAgents: 5,
  priority: "critical"
})

// Step 4: Monitor deployment
mcp__claude-flow__swarm_status({ verbose: true })
```

### 5. Workflow Automation Skill

**MCP Tool Usage:**

```javascript
// Step 1: Initialize mesh swarm for collaborative workflow development
mcp__claude-flow__swarm_init({
  topology: "mesh",
  maxAgents: 6,
  strategy: "balanced"
})

// Step 2: Spawn workflow specialists
mcp__claude-flow__agent_spawn({ type: "coder", name: "cicd-engineer" })
mcp__claude-flow__agent_spawn({ type: "coordinator", name: "workflow-automation" })
mcp__claude-flow__agent_spawn({ type: "researcher", name: "tester" })
mcp__claude-flow__agent_spawn({ type: "analyst", name: "security-auditor" })
mcp__claude-flow__agent_spawn({ type: "optimizer", name: "perf-analyzer" })

// Step 3 (Optional): Monitor workflow execution in real-time
mcp__flow-nexus__execution_stream_subscribe({
  stream_type: "github-integration",
  sandbox_id: "sandbox-123"
})

// Step 4: Get agent performance metrics
mcp__claude-flow__agent_metrics({ metric: "performance" })
```

## Memory Coordination

All agents coordinate through Claude Flow's memory system using hooks:

### Pre-Task Hooks

```bash
# Restore shared context before agent starts work
npx claude-flow@alpha hooks session-restore --session-id "swarm-${SWARM_ID}"

# Prepare agent for task
npx claude-flow@alpha hooks pre-task --description "security review of PR #123"
```

### During Task Hooks

```bash
# Store agent findings in shared memory
npx claude-flow@alpha hooks post-edit \
  --file "security-findings.json" \
  --memory-key "github/pr-review/security"

# Notify other agents of progress
npx claude-flow@alpha hooks notify \
  --message "Security scan complete: 2 high-severity issues found"
```

### Post-Task Hooks

```bash
# Mark task complete and update metrics
npx claude-flow@alpha hooks post-task --task-id "task-123"

# Export session metrics
npx claude-flow@alpha hooks session-end --export-metrics true
```

## Error Handling

### MCP Connection Errors

```javascript
// Check if MCP server is running
try {
  await mcp__claude-flow__swarm_status()
} catch (error) {
  console.error("Claude Flow MCP server not available")
  console.log("Run: claude mcp add claude-flow npx claude-flow@alpha mcp start")
}
```

### Rate Limiting

```javascript
// Claude Flow has no rate limits (local execution)
// Flow-Nexus respects GitHub API rate limits

// For GitHub API operations, implement exponential backoff
let retries = 0
const maxRetries = 3

while (retries < maxRetries) {
  try {
    await github_operation()
    break
  } catch (error) {
    if (error.status === 429) {  // Rate limited
      const delay = Math.pow(2, retries) * 1000
      await sleep(delay)
      retries++
    } else {
      throw error
    }
  }
}
```

### Swarm Recovery

```javascript
// If swarm crashes, check status and reinitialize if needed
const status = await mcp__claude-flow__swarm_status()

if (status.state === "failed" || !status.active) {
  console.log("Swarm failed, reinitializing...")

  await mcp__claude-flow__swarm_init({
    topology: "mesh",
    maxAgents: 5,
    strategy: "balanced"
  })
}
```

## Best Practices

### 1. Topology Selection

Choose topology based on coordination pattern:

- **Mesh**: When agents need peer-to-peer communication (code review)
- **Hierarchical**: When centralized decision-making required (multi-repo, releases)
- **Star**: When hub-and-spoke coordination optimal (project management)
- **Ring**: When sequential processing with feedback loops needed

### 2. Agent Scaling

Start with minimal agents and scale up:

```javascript
// Start small
mcp__claude-flow__swarm_init({ maxAgents: 3 })

// Scale up based on task complexity
mcp__claude-flow__swarm_scale({ target_agents: 8 })
```

### 3. Memory Keys

Use consistent memory key naming:

```
github/<skill>/<category>/<identifier>

Examples:
- github/pr-review/security/findings
- github/multi-repo/dependencies/graph
- github/project/sprint/capacity
- github/release/v2.0.0/artifacts
- github/workflow/ci-pipeline/metrics
```

### 4. Monitoring

Always monitor swarm health:

```javascript
// Check swarm status before critical operations
const status = await mcp__claude-flow__swarm_status({ verbose: true })

if (status.health !== "healthy") {
  console.warn("Swarm health degraded:", status.issues)
}

// Monitor agent metrics periodically
setInterval(async () => {
  const metrics = await mcp__claude-flow__agent_metrics({ metric: "all" })
  console.log("Swarm metrics:", metrics)
}, 60000)  // Every minute
```

### 5. Cleanup

Destroy swarms after completion to free resources:

```javascript
// After task completion
await mcp__claude-flow__task_results({ taskId: "task-123" })

// Clean up swarm
await mcp__claude-flow__swarm_destroy({ swarm_id: "swarm-456" })
```

## Troubleshooting

### MCP Server Not Responding

```bash
# Check MCP server status
claude mcp list

# Restart MCP server
claude mcp restart claude-flow

# View MCP logs
claude mcp logs claude-flow
```

### Agent Spawn Failures

```bash
# Check available agent capacity
npx claude-flow@alpha swarm status --verbose

# Reduce concurrent agents if hitting limits
mcp__claude-flow__swarm_init({ maxAgents: 3 })  # Instead of 10
```

### Memory Coordination Issues

```bash
# Debug memory keys
npx claude-flow@alpha memory list

# Retrieve specific memory
npx claude-flow@alpha memory retrieve --key "github/pr-review/*"

# Clear stale memory
npx claude-flow@alpha memory clear --pattern "github/old-project/*"
```

### Flow-Nexus Authentication

```bash
# Check auth status
npx flow-nexus@latest auth status

# Re-authenticate if token expired
npx flow-nexus@latest login

# Check credit balance (for paid features)
npx flow-nexus@latest balance
```

## Advanced Features

### Neural Training

Train patterns from successful operations:

```bash
# After successful PR review
npx claude-flow@alpha hooks post-task \
  --task-id "pr-review-123" \
  --train-patterns true
```

### Performance Benchmarking

```javascript
// Benchmark swarm performance
mcp__claude-flow__benchmark_run({
  type: "swarm",
  iterations: 10
})
```

### Custom Agent Capabilities

```javascript
// Spawn agent with custom capabilities
mcp__claude-flow__agent_spawn({
  type: "analyst",
  name: "custom-reviewer",
  capabilities: [
    "rust-expertise",
    "async-patterns",
    "zero-copy-optimization"
  ]
})
```

## References

- Claude Flow Documentation: https://github.com/ruvnet/claude-flow
- Flow-Nexus Platform: https://flow-nexus.ruv.io
- MCP Protocol Specification: https://modelcontextprotocol.io
- GitHub REST API: https://docs.github.com/rest
- GitHub GraphQL API: https://docs.github.com/graphql
