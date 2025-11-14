# MCP Tools Quick Reference

**Version**: 1.0.0
**Created**: 2025-11-01
**Purpose**: Quick lookup guide for MCP tools by use case

---

## Quick Tool Finder

### I need to...

#### ...coordinate multiple agents
- `mcp__claude-flow__swarm_init` - Initialize swarm
- `mcp__claude-flow__agent_spawn` - Spawn agent types
- `mcp__claude-flow__task_orchestrate` - Delegate tasks
- `mcp__ruv-swarm__daa_init` - Autonomous agents

#### ...remember across sessions
- `mcp__memory-mcp__memory_store` - Store information
- `mcp__memory-mcp__vector_search` - Search memories
- `mcp__ruv-swarm__daa_knowledge_share` - Share knowledge

#### ...check code quality
- `mcp__connascence-analyzer__analyze_file` - Single file
- `mcp__connascence-analyzer__analyze_workspace` - Entire project
- `mcp__focused-changes__analyze_changes` - Validate changes

#### ...run code safely
- `mcp__flow-nexus__sandbox_create` - Create environment
- `mcp__flow-nexus__sandbox_execute` - Run code
- `mcp__flow-nexus__sandbox_logs` - Check output

#### ...train ML models
- `mcp__flow-nexus__neural_train` - Single instance
- `mcp__flow-nexus__neural_cluster_init` - Distributed
- `mcp__flow-nexus__neural_train_distributed` - Train at scale

#### ...automate browser testing
- `mcp__playwright__browser_navigate` - Open page
- `mcp__playwright__browser_snapshot` - Capture state
- `mcp__playwright__browser_click` - Interact
- `mcp__playwright__browser_evaluate` - Run JavaScript

#### ...manage files
- `mcp__filesystem__read_text_file` - Read file
- `mcp__filesystem__write_file` - Write file
- `mcp__filesystem__edit_file` - Edit file
- `mcp__filesystem__directory_tree` - View structure

#### ...authorize payments
- `mcp__agentic-payments__create_active_mandate` - Set limits
- `mcp__agentic-payments__sign_mandate` - Authorize
- `mcp__agentic-payments__verify_mandate` - Validate

#### ...analyze GitHub repos
- `mcp__flow-nexus__github_repo_analyze` - Analyze repo
- `mcp__flow-nexus__workflow_create` - Automate workflows

#### ...reason through problems
- `mcp__sequential-thinking__sequentialthinking` - Step-by-step reasoning

---

## Tool Categories Cheat Sheet

### Core (Required) - 18 tools
**Server**: claude-flow
**Use**: Basic coordination
**Access**: ALL agents

### Memory - 2 tools
**Server**: memory-mcp
**Use**: Persistent context
**Access**: ALL agents

### Code Quality - 6 tools
**Servers**: connascence-analyzer, focused-changes
**Use**: Quality checks
**Access**: 14 code quality agents

### Cloud Execution - 27 tools
**Server**: flow-nexus
**Use**: Sandboxes, deployment
**Access**: Development agents
**Auth**: Required

### Neural Networks - 22 tools
**Server**: flow-nexus
**Use**: ML training
**Access**: ML developers
**Auth**: Required

### Browser Automation - 17 tools
**Server**: playwright
**Use**: E2E testing
**Access**: Testing agents

### File Operations - 12 tools
**Server**: filesystem
**Use**: File management
**Access**: ALL agents

### Payments - 14 tools
**Server**: agentic-payments
**Use**: Payment auth
**Access**: Payment agents

### DAA - 12 tools
**Server**: ruv-swarm
**Use**: Autonomous agents
**Access**: Advanced coordinators

---

## Agent-to-Tools Quick Map

### Coder
- Universal (18) + Memory (2) + Connascence (3) + Sandbox (9) + Files (12) = **44 tools**

### Reviewer
- Universal (18) + Memory (2) + Connascence (3) + Files (12) = **35 tools**

### Tester
- Universal (18) + Memory (2) + Connascence (3) + Sandbox (9) + Browser (17) + Files (12) = **61 tools**

### ML Developer
- Universal (18) + Memory (2) + Connascence (3) + Sandbox (9) + Neural (22) + Files (12) = **66 tools**

### Backend Dev
- Universal (18) + Memory (2) + Connascence (3) + Sandbox (9) + Files (12) = **44 tools**

### DevOps
- Universal (18) + Memory (2) + Sandbox (9) + Workflow (6) + Files (12) = **47 tools**

### Security Manager
- Universal (18) + Memory (2) + Payments (14) + Files (12) = **46 tools**

### Coordinator
- Universal (18) + Memory (2) + DAA (12) = **32 tools**

---

## Common Workflows

### 1. Code → Test → Deploy
```javascript
sandbox_create → write_file → analyze_file → sandbox_execute →
workflow_create → workflow_execute → storage_upload → system_health
```

### 2. Research → Plan → Implement
```javascript
vector_search → sequentialthinking → memory_store →
write_file → analyze_file → sandbox_execute → memory_store
```

### 3. Train Model → Validate → Deploy
```javascript
neural_list_templates → neural_cluster_init → neural_train_distributed →
neural_performance_benchmark → neural_validation_workflow → neural_publish_template
```

### 4. Review PR → Test → Merge
```javascript
github_repo_analyze → read_multiple_files → analyze_file →
sandbox_execute → memory_store → workflow_create
```

### 5. Browser Test → Validate → Report
```javascript
browser_navigate → browser_snapshot → browser_fill_form →
browser_click → browser_console_messages → browser_take_screenshot → memory_store
```

---

## Installation Quick Start

```bash
# 1. Core (required)
claude mcp add claude-flow npx claude-flow@alpha mcp start

# 2. Enhanced (optional)
claude mcp add ruv-swarm npx ruv-swarm mcp start

# 3. Cloud (optional, requires auth)
claude mcp add flow-nexus npx flow-nexus@latest mcp start
npx flow-nexus@latest register
npx flow-nexus@latest login

# 4. Verify
claude mcp list
```

---

## Troubleshooting Quick Fixes

### Server not connected
```bash
claude mcp remove <server>
claude mcp add <server> <command>
```

### Authentication failed
```bash
npx flow-nexus@latest logout
npx flow-nexus@latest login
```

### Connascence errors
```bash
cd C:\Users\17175\Desktop\connascence
.\venv-connascence\Scripts\python.exe mcp/cli.py health-check
```

### Memory MCP errors
```bash
cd C:\Users\17175\Desktop\memory-mcp-triple-system
.\venv-memory\Scripts\python.exe -m pytest tests/
```

---

## Tool Limits & Quotas

### Memory MCP
- Search limit: 5-20 results (default: 5)
- Retention: 24h/7d/30d+ (automatic)
- Vector dims: 384
- No quota limits

### Connascence
- Analysis speed: ~0.018s per file
- Detects: 7+ violation types
- No quota limits (local)

### Flow-Nexus (requires credits)
- Sandboxes: Timeout 3600s default
- Neural training: Tier-based (nano/mini/small/medium/large)
- Storage: Bucket-based
- Check balance: `check_balance`

### Playwright
- Screenshot types: png/jpeg
- Network requests: All captured
- Console messages: All captured
- No quota limits

---

**Full Documentation**: See `MCP-TOOLS-INVENTORY.md` for complete details

**Agent Assignments**: See `MCP-TOOL-TO-AGENT-ASSIGNMENTS.md` for all 86 agents

**Integration Guide**: See `integration-plans/MCP-INTEGRATION-GUIDE.md` for setup
