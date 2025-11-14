# MCP Tools Inventory - All Servers

**Date**: 2025-10-29
**Sources**: ruv-swarm, flow-nexus, agentic-payments, claude-flow@alpha
**Total Tools**: 140+

---

## MCP Server Status

| Server | Status | Tools Available | Connection |
|--------|--------|-----------------|------------|
| ruv-swarm | ✅ Connected | 35 tools | Active |
| flow-nexus | ✅ Connected | 70+ tools | Active |
| agentic-payments | ✅ Connected | 0 tools (resources only) | Active |
| claude-flow@alpha | ❌ Not connected | ~50 tools (estimated) | Failed |

---

## ruv-swarm MCP Tools (35 tools)

**Category: Swarm Coordination (NO TIMEOUT VERSION)**

### Core Swarm Management (3 tools)
1. `mcp__ruv-swarm__swarm_init` - Initialize swarm topology (mesh, hierarchical, ring, star)
2. `mcp__ruv-swarm__swarm_status` - Get swarm status and agent information
3. `mcp__ruv-swarm__swarm_monitor` - Real-time swarm activity monitoring

### Agent Management (3 tools)
4. `mcp__ruv-swarm__agent_spawn` - Spawn agents (researcher, coder, analyst, optimizer, coordinator)
5. `mcp__ruv-swarm__agent_list` - List active agents with filtering
6. `mcp__ruv-swarm__agent_metrics` - Get agent performance metrics

### Task Orchestration (3 tools)
7. `mcp__ruv-swarm__task_orchestrate` - Orchestrate tasks across swarm
8. `mcp__ruv-swarm__task_status` - Check running task progress
9. `mcp__ruv-swarm__task_results` - Retrieve completed task results

### Performance & Features (3 tools)
10. `mcp__ruv-swarm__benchmark_run` - Execute performance benchmarks
11. `mcp__ruv-swarm__features_detect` - Detect runtime capabilities
12. `mcp__ruv-swarm__memory_usage` - Get memory usage statistics

### Neural & Learning (3 tools)
13. `mcp__ruv-swarm__neural_status` - Get neural agent status and metrics
14. `mcp__ruv-swarm__neural_train` - Train neural agents with sample tasks
15. `mcp__ruv-swarm__neural_patterns` - Get cognitive pattern information

### DAA (Decentralized Autonomous Agents) (12 tools)
16. `mcp__ruv-swarm__daa_init` - Initialize DAA service
17. `mcp__ruv-swarm__daa_agent_create` - Create autonomous agent
18. `mcp__ruv-swarm__daa_agent_adapt` - Trigger agent adaptation
19. `mcp__ruv-swarm__daa_workflow_create` - Create autonomous workflow
20. `mcp__ruv-swarm__daa_workflow_execute` - Execute DAA workflow
21. `mcp__ruv-swarm__daa_knowledge_share` - Share knowledge between agents
22. `mcp__ruv-swarm__daa_learning_status` - Get learning progress
23. `mcp__ruv-swarm__daa_cognitive_pattern` - Analyze/change cognitive patterns
24. `mcp__ruv-swarm__daa_meta_learning` - Enable meta-learning capabilities
25. `mcp__ruv-swarm__daa_performance_metrics` - Get comprehensive DAA metrics

**Usage Pattern**:
```javascript
// Initialize swarm with mesh topology
mcp__ruv-swarm__swarm_init({
  topology: "mesh",
  maxAgents: 8,
  strategy: "adaptive"
})

// Spawn specialist agents
mcp__ruv-swarm__agent_spawn({
  type: "researcher",
  capabilities: ["analysis", "synthesis"]
})

// Orchestrate complex task
mcp__ruv-swarm__task_orchestrate({
  task: "Complete system analysis",
  strategy: "parallel",
  priority: "high"
})
```

---

## flow-nexus MCP Tools (70+ tools)

**Category: Cloud-Based Orchestration & Services**

### Swarm & Agent Management (9 tools)
1. `mcp__flow-nexus__swarm_init` - Initialize cloud-based swarm
2. `mcp__flow-nexus__swarm_list` - List active swarms
3. `mcp__flow-nexus__swarm_status` - Get swarm status
4. `mcp__flow-nexus__swarm_scale` - Scale swarm up/down
5. `mcp__flow-nexus__swarm_destroy` - Terminate swarm
6. `mcp__flow-nexus__swarm_create_from_template` - Create from template
7. `mcp__flow-nexus__swarm_templates_list` - List available templates
8. `mcp__flow-nexus__agent_spawn` - Spawn cloud agent
9. `mcp__flow-nexus__task_orchestrate` - Cloud task orchestration

### Neural Networks & AI (16 tools)
10. `mcp__flow-nexus__neural_train` - Train neural network with custom config
11. `mcp__flow-nexus__neural_predict` - Run inference on trained model
12. `mcp__flow-nexus__neural_list_templates` - List neural network templates
13. `mcp__flow-nexus__neural_deploy_template` - Deploy neural template
14. `mcp__flow-nexus__neural_training_status` - Check training job status
15. `mcp__flow-nexus__neural_list_models` - List user's trained models
16. `mcp__flow-nexus__neural_validation_workflow` - Create validation workflow
17. `mcp__flow-nexus__neural_publish_template` - Publish model as template
18. `mcp__flow-nexus__neural_rate_template` - Rate a template
19. `mcp__flow-nexus__neural_performance_benchmark` - Run performance benchmarks
20. `mcp__flow-nexus__neural_cluster_init` - Initialize distributed neural cluster
21. `mcp__flow-nexus__neural_node_deploy` - Deploy neural node in E2B sandbox
22. `mcp__flow-nexus__neural_cluster_connect` - Connect cluster nodes
23. `mcp__flow-nexus__neural_train_distributed` - Distributed training
24. `mcp__flow-nexus__neural_cluster_status` - Get cluster status
25. `mcp__flow-nexus__neural_predict_distributed` - Distributed inference

### Sandbox Management (9 tools)
26. `mcp__flow-nexus__sandbox_create` - Create code execution sandbox
27. `mcp__flow-nexus__sandbox_execute` - Execute code in sandbox
28. `mcp__flow-nexus__sandbox_list` - List all sandboxes
29. `mcp__flow-nexus__sandbox_stop` - Stop running sandbox
30. `mcp__flow-nexus__sandbox_configure` - Configure sandbox environment
31. `mcp__flow-nexus__sandbox_delete` - Delete sandbox
32. `mcp__flow-nexus__sandbox_status` - Get sandbox status
33. `mcp__flow-nexus__sandbox_upload` - Upload file to sandbox
34. `mcp__flow-nexus__sandbox_logs` - Get sandbox logs

### Templates & App Store (8 tools)
35. `mcp__flow-nexus__template_list` - List deployment templates
36. `mcp__flow-nexus__template_get` - Get template details
37. `mcp__flow-nexus__template_deploy` - Deploy template with variables
38. `mcp__flow-nexus__app_store_list_templates` - List app templates
39. `mcp__flow-nexus__app_store_publish_app` - Publish app to store
40. `mcp__flow-nexus__app_get` - Get app details
41. `mcp__flow-nexus__app_update` - Update app information
42. `mcp__flow-nexus__app_search` - Search apps with filters

### Challenges & Gamification (6 tools)
43. `mcp__flow-nexus__challenges_list` - List available challenges
44. `mcp__flow-nexus__challenge_get` - Get challenge details
45. `mcp__flow-nexus__challenge_submit` - Submit challenge solution
46. `mcp__flow-nexus__app_store_complete_challenge` - Mark challenge complete
47. `mcp__flow-nexus__leaderboard_get` - Get leaderboard rankings
48. `mcp__flow-nexus__achievements_list` - List user achievements

### Authentication & User Management (10 tools)
49. `mcp__flow-nexus__auth_status` - Check auth status
50. `mcp__flow-nexus__auth_init` - Initialize authentication
51. `mcp__flow-nexus__user_register` - Register new user
52. `mcp__flow-nexus__user_login` - Login user
53. `mcp__flow-nexus__user_logout` - Logout user
54. `mcp__flow-nexus__user_verify_email` - Verify email with token
55. `mcp__flow-nexus__user_reset_password` - Request password reset
56. `mcp__flow-nexus__user_update_password` - Update password
57. `mcp__flow-nexus__user_upgrade` - Upgrade user tier
58. `mcp__flow-nexus__user_stats` - Get user statistics

### Payments & Credits (7 tools)
59. `mcp__flow-nexus__app_store_earn_ruv` - Award rUv credits
60. `mcp__flow-nexus__ruv_balance` - Get credit balance
61. `mcp__flow-nexus__ruv_history` - Get transaction history
62. `mcp__flow-nexus__check_balance` - Check current balance
63. `mcp__flow-nexus__create_payment_link` - Create payment link
64. `mcp__flow-nexus__configure_auto_refill` - Configure auto-refill
65. `mcp__flow-nexus__get_payment_history` - Get payment history

### Workflow & Execution (6 tools)
66. `mcp__flow-nexus__workflow_create` - Create event-driven workflow
67. `mcp__flow-nexus__workflow_execute` - Execute workflow
68. `mcp__flow-nexus__workflow_status` - Get workflow status
69. `mcp__flow-nexus__workflow_list` - List workflows
70. `mcp__flow-nexus__workflow_agent_assign` - Assign optimal agent
71. `mcp__flow-nexus__workflow_queue_status` - Check message queue

### Real-time & Storage (10 tools)
72. `mcp__flow-nexus__execution_stream_subscribe` - Subscribe to execution streams
73. `mcp__flow-nexus__execution_stream_status` - Get stream status
74. `mcp__flow-nexus__execution_files_list` - List execution files
75. `mcp__flow-nexus__execution_file_get` - Get file content
76. `mcp__flow-nexus__realtime_subscribe` - Subscribe to database changes
77. `mcp__flow-nexus__realtime_unsubscribe` - Unsubscribe from changes
78. `mcp__flow-nexus__realtime_list` - List active subscriptions
79. `mcp__flow-nexus__storage_upload` - Upload file to storage
80. `mcp__flow-nexus__storage_delete` - Delete file from storage
81. `mcp__flow-nexus__storage_list` - List files in bucket

### System & Analytics (5 tools)
82. `mcp__flow-nexus__system_health` - Check system health
83. `mcp__flow-nexus__audit_log` - Get audit log entries
84. `mcp__flow-nexus__market_data` - Get market statistics
85. `mcp__flow-nexus__app_analytics` - Get app analytics
86. `mcp__flow-nexus__seraphina_chat` - Chat with Queen Seraphina AI

### GitHub Integration (2 tools)
87. `mcp__flow-nexus__github_repo_analyze` - Analyze GitHub repository
88. `mcp__flow-nexus__daa_agent_create` - Create DAA agent (duplicate functionality)

**Usage Pattern**:
```javascript
// Create cloud sandbox
mcp__flow-nexus__sandbox_create({
  template: "nodejs",
  env_vars: { API_KEY: "..." },
  anthropic_key: "..."
})

// Train neural network
mcp__flow-nexus__neural_train({
  config: {
    architecture: { type: "transformer", layers: [...] },
    training: { epochs: 100, batch_size: 32 }
  },
  tier: "medium"
})

// Deploy from template
mcp__flow-nexus__template_deploy({
  template_name: "rest-api-starter",
  variables: { port: 3000, db: "postgresql" }
})
```

---

## claude-flow@alpha MCP Tools (Estimated ~50 tools)

**Status**: ❌ Not connected (connection failed)
**Expected Tools** (based on usage patterns):

### Coordination & Memory (Estimated)
- `mcp__claude-flow__swarm_init`
- `mcp__claude-flow__agent_spawn`
- `mcp__claude-flow__task_orchestrate`
- `mcp__claude-flow__memory_store`
- `mcp__claude-flow__memory_retrieve`
- `mcp__claude-flow__memory_search`
- `mcp__claude-flow__agent_delegate`
- `mcp__claude-flow__agent_escalate`
- `mcp__claude-flow__neural_train`
- `mcp__claude-flow__hooks_pre_task`
- `mcp__claude-flow__hooks_post_task`
- `mcp__claude-flow__hooks_session_restore`
- `mcp__claude-flow__hooks_session_end`

**Note**: Exact tool list unavailable due to connection failure. Tools inferred from CLAUDE.md documentation and usage patterns.

---

## agentic-payments MCP

**Status**: ✅ Connected
**Tools**: 0 (resources only, not tools)
**Resources**: Payment/billing management (implementation unclear)

---

## Tool Categorization for Agent Use

### Universal Tools (Available to ALL agents)

**Coordination Tools**:
- swarm_init, agent_spawn, task_orchestrate
- agent_delegate, agent_escalate, agent_list
- swarm_status, swarm_monitor

**Memory Tools**:
- memory_store, memory_retrieve, memory_search
- memory_usage, daa_knowledge_share

**Monitoring Tools**:
- task_status, task_results, agent_metrics
- performance_metrics, benchmark_run

### Specialist Tools (Role-specific)

**Neural/ML Specialists**:
- neural_train, neural_predict, neural_deploy_template
- neural_cluster_init, neural_train_distributed
- neural_performance_benchmark, neural_validation_workflow

**DevOps/Infrastructure Specialists**:
- sandbox_create, sandbox_execute, sandbox_configure
- template_deploy, workflow_create, workflow_execute
- system_health, audit_log

**Data/Analytics Specialists**:
- neural_list_templates, neural_list_models
- app_analytics, market_data, leaderboard_get

**Business/Product Specialists**:
- app_store_publish_app, challenges_list, challenge_submit
- ruv_balance, create_payment_link, achievements_list

**Security/Auth Specialists**:
- auth_init, user_verify_email, user_update_password
- audit_log, system_health

**GitHub/Repository Specialists**:
- github_repo_analyze

---

## MCP Tool Usage Patterns

### Pattern 1: Agent Coordination

```javascript
// 1. Initialize swarm
mcp__ruv-swarm__swarm_init({ topology: "mesh", maxAgents: 8 })

// 2. Spawn specialist agents
mcp__ruv-swarm__agent_spawn({ type: "researcher", name: "data-analyst" })
mcp__ruv-swarm__agent_spawn({ type: "coder", name: "backend-dev" })

// 3. Orchestrate task
mcp__ruv-swarm__task_orchestrate({
  task: "Build REST API",
  strategy: "adaptive",
  maxAgents: 2
})

// 4. Monitor progress
mcp__ruv-swarm__task_status({ taskId: "task-123" })

// 5. Get results
mcp__ruv-swarm__task_results({ taskId: "task-123", format: "detailed" })
```

### Pattern 2: Memory Coordination

```javascript
// Store data for other agents
mcp__claude-flow__memory_store({
  key: "backend-dev/api-v2/schema-design",
  value: { tables: [...], relations: [...] },
  ttl: 86400
})

// Retrieve in another agent
mcp__claude-flow__memory_retrieve({
  key: "backend-dev/api-v2/schema-design"
})

// Search across namespaces
mcp__claude-flow__memory_search({
  pattern: "backend-dev/api-v2/*"
})
```

### Pattern 3: Neural Training

```javascript
// 1. Create training cluster
mcp__flow-nexus__neural_cluster_init({
  name: "ml-training-cluster",
  architecture: "transformer",
  topology: "mesh"
})

// 2. Deploy worker nodes
mcp__flow-nexus__neural_node_deploy({
  cluster_id: "cluster-123",
  model: "large",
  role: "worker"
})

// 3. Start distributed training
mcp__flow-nexus__neural_train_distributed({
  cluster_id: "cluster-123",
  dataset: "training-data.json",
  epochs: 100,
  federated: true
})

// 4. Monitor training
mcp__flow-nexus__neural_cluster_status({ cluster_id: "cluster-123" })
```

### Pattern 4: Cloud Deployment

```javascript
// 1. Create sandbox
const sandbox = await mcp__flow-nexus__sandbox_create({
  template: "nodejs",
  env_vars: { NODE_ENV: "production" },
  anthropic_key: process.env.ANTHROPIC_KEY
})

// 2. Upload code
await mcp__flow-nexus__sandbox_upload({
  sandbox_id: sandbox.id,
  file_path: "app.js",
  content: codeContent
})

// 3. Execute
await mcp__flow-nexus__sandbox_execute({
  sandbox_id: sandbox.id,
  code: "node app.js",
  timeout: 60000
})

// 4. Get logs
await mcp__flow-nexus__sandbox_logs({
  sandbox_id: sandbox.id,
  lines: 100
})
```

---

## Tool Mapping to 90 Agents

### Core Development Agents (5)
- **coder**: sandbox_execute, template_deploy, workflow_create
- **reviewer**: agent_metrics, audit_log, performance_metrics
- **tester**: benchmark_run, sandbox_execute, neural_validation_workflow
- **planner**: task_orchestrate, workflow_create, daa_workflow_create
- **researcher**: neural_list_templates, app_search, market_data

### Swarm Coordination Agents (5)
- **hierarchical-coordinator**: swarm_init (hierarchical), agent_spawn, task_orchestrate
- **mesh-coordinator**: swarm_init (mesh), daa_agent_create, daa_workflow_execute
- **adaptive-coordinator**: swarm_init (adaptive), daa_cognitive_pattern, daa_meta_learning
- **collective-intelligence-coordinator**: daa_knowledge_share, daa_learning_status
- **swarm-memory-manager**: memory_store, memory_retrieve, memory_search

### Neural & ML Agents (1)
- **safla-neural**: All neural_* tools, neural_cluster_*, daa_meta_learning

### Flow Nexus Platform Agents (9)
- **flow-nexus-swarm**: All flow-nexus swarm_* tools
- **flow-nexus-sandbox**: All sandbox_* tools
- **flow-nexus-neural**: All neural_* tools
- **flow-nexus-workflow**: All workflow_* tools
- **flow-nexus-auth**: All auth_* and user_* tools
- **flow-nexus-app-store**: All app_* and template_* tools
- **flow-nexus-challenges**: All challenge_* and achievement_* tools
- **flow-nexus-payments**: All ruv_* and payment_* tools
- **flow-nexus-user-tools**: user_profile, user_update_profile, storage_* tools

### GitHub Agents (13)
- **pr-manager**: github_repo_analyze
- **code-review-swarm**: agent_spawn, task_orchestrate, github_repo_analyze
- [Others use combination of coordination + GitHub tools]

---

## Next Steps

1. ✅ Complete MCP tools inventory (this document)
2. ⏳ Create detailed skill-to-agent mapping matrix
3. ⏳ Store universal tool patterns in memory
4. ⏳ Create SOP skills that use specific tool sequences
5. ⏳ Begin agent rewrites with exact MCP tool specifications

---

**Status**: MCP tools inventory complete
**Total Tools**: 140+ across 4 servers (3 connected)
**Ready for**: Skill-to-agent mapping and agent rewrites
