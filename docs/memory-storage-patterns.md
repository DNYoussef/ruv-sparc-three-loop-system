# Memory Storage Patterns for Universal Agent Access

**Date**: 2025-10-29
**Purpose**: Persistent memory patterns for all 90 agents
**MCP Tools**: claude-flow memory_store, ruv-swarm daa_knowledge_share

---

## Memory Namespace Convention

**Format**: `{category}/{agent-role}/{task-id}/{data-type}`

**Examples**:
- `universal-commands/file-operations/read`
- `universal-commands/git-operations/commit`
- `universal-tools/mcp/swarm_init`
- `skill-mappings/backend-dev/skills`
- `agent-tasks/marketing-specialist/campaign-123/results`

---

## Universal Commands Storage

### File Operations (8 commands)

```javascript
// Store in memory namespace: universal-commands/file-operations/*

{
  "file-read": {
    "command": "/file-read --path {file} --format {format}",
    "when": "Need to read file contents",
    "usage": "Read configuration, code, data files",
    "examples": [
      "/file-read --path package.json --format json",
      "/file-read --path README.md --format markdown"
    ],
    "available_to": "ALL agents"
  },

  "file-write": {
    "command": "/file-write --path {file} --content {content}",
    "when": "Need to create or update files",
    "usage": "Write code, config, documentation",
    "examples": [
      "/file-write --path src/app.js --content {code}",
      "/file-write --path docs/API.md --content {docs}"
    ],
    "available_to": "ALL agents"
  },

  "file-edit": {
    "command": "/file-edit --path {file} --old {old} --new {new}",
    "when": "Need to modify existing file content",
    "usage": "Update specific sections without full rewrite",
    "examples": [
      "/file-edit --path config.js --old 'port: 3000' --new 'port: 8080'"
    ],
    "available_to": "ALL agents"
  },

  "glob-search": {
    "command": "/glob-search --pattern {pattern}",
    "when": "Need to find files by pattern",
    "usage": "Locate files matching glob patterns",
    "examples": [
      "/glob-search --pattern '**/*.js'",
      "/glob-search --pattern 'src/components/*.tsx'"
    ],
    "available_to": "ALL agents"
  },

  "grep-search": {
    "command": "/grep-search --pattern {pattern} --path {path}",
    "when": "Need to search file contents",
    "usage": "Find text patterns in files",
    "examples": [
      "/grep-search --pattern 'TODO' --path src/",
      "/grep-search --pattern 'function.*async' --path ."
    ],
    "available_to": "ALL agents"
  },

  "file-delete": {
    "command": "/file-delete --path {file}",
    "when": "Need to remove files",
    "usage": "Clean up temporary or obsolete files",
    "examples": [
      "/file-delete --path temp/cache.json"
    ],
    "available_to": "ALL agents"
  },

  "directory-create": {
    "command": "/directory-create --path {dir}",
    "when": "Need to create directory structure",
    "usage": "Set up project folders",
    "examples": [
      "/directory-create --path src/components",
      "/directory-create --path tests/integration"
    ],
    "available_to": "ALL agents"
  },

  "directory-list": {
    "command": "/directory-list --path {dir}",
    "when": "Need to see directory contents",
    "usage": "Explore file structure",
    "examples": [
      "/directory-list --path src/",
      "/directory-list --path ."
    ],
    "available_to": "ALL agents"
  }
}
```

### Git Operations (10 commands)

```javascript
// Store in memory namespace: universal-commands/git-operations/*

{
  "git-status": {
    "command": "/git-status",
    "when": "Need to check repository state",
    "usage": "See modified, staged, untracked files",
    "examples": ["/git-status"],
    "available_to": "ALL agents"
  },

  "git-diff": {
    "command": "/git-diff --file {file}",
    "when": "Need to see changes",
    "usage": "Review modifications before commit",
    "examples": [
      "/git-diff --file src/app.js",
      "/git-diff"
    ],
    "available_to": "ALL agents"
  },

  "git-log": {
    "command": "/git-log --limit {n}",
    "when": "Need to see commit history",
    "usage": "Review recent commits, find patterns",
    "examples": [
      "/git-log --limit 10",
      "/git-log --author {name}"
    ],
    "available_to": "ALL agents"
  },

  "git-add": {
    "command": "/git-add --file {file}",
    "when": "Need to stage files for commit",
    "usage": "Prepare files for commit",
    "examples": [
      "/git-add --file src/app.js",
      "/git-add ."
    ],
    "available_to": "ALL agents"
  },

  "git-commit": {
    "command": "/git-commit --message {message}",
    "when": "Need to save changes",
    "usage": "Create commit with descriptive message",
    "examples": [
      "/git-commit --message 'feat: Add authentication module'",
      "/git-commit --message 'fix: Resolve memory leak in cache'"
    ],
    "pattern": "Follow conventional commits (feat, fix, docs, style, refactor, test, chore)",
    "available_to": "ALL agents"
  },

  "git-push": {
    "command": "/git-push",
    "when": "Need to sync with remote",
    "usage": "Push commits to remote repository",
    "examples": ["/git-push"],
    "available_to": "ALL agents"
  },

  "git-branch": {
    "command": "/git-branch --name {branch}",
    "when": "Need to create or list branches",
    "usage": "Manage branches",
    "examples": [
      "/git-branch",
      "/git-branch --name feature/auth"
    ],
    "available_to": "ALL agents"
  },

  "git-checkout": {
    "command": "/git-checkout --branch {branch}",
    "when": "Need to switch branches",
    "usage": "Change working branch",
    "examples": [
      "/git-checkout --branch main",
      "/git-checkout --branch feature/auth"
    ],
    "available_to": "ALL agents"
  },

  "git-merge": {
    "command": "/git-merge --branch {branch}",
    "when": "Need to merge branches",
    "usage": "Integrate changes from another branch",
    "examples": [
      "/git-merge --branch feature/auth"
    ],
    "available_to": "ALL agents"
  },

  "git-stash": {
    "command": "/git-stash",
    "when": "Need to save work temporarily",
    "usage": "Stash uncommitted changes",
    "examples": [
      "/git-stash",
      "/git-stash pop"
    ],
    "available_to": "ALL agents"
  }
}
```

### Communication & Coordination (8 commands)

```javascript
// Store in memory namespace: universal-commands/communication/*

{
  "communicate-notify": {
    "command": "/communicate-notify --to {recipient} --message {message}",
    "when": "Need to notify other agents or users",
    "usage": "Send notifications about task completion or issues",
    "examples": [
      "/communicate-notify --to frontend-dev --message 'API endpoints ready'",
      "/communicate-notify --to user --message 'Build completed successfully'"
    ],
    "available_to": "ALL agents"
  },

  "communicate-report": {
    "command": "/communicate-report --type {type} --data {data}",
    "when": "Need to generate status reports",
    "usage": "Create structured reports",
    "examples": [
      "/communicate-report --type progress --data {...}",
      "/communicate-report --type completion --data {...}"
    ],
    "available_to": "ALL agents"
  },

  "communicate-log": {
    "command": "/communicate-log --level {level} --message {message}",
    "when": "Need to log events or information",
    "usage": "Track execution progress",
    "examples": [
      "/communicate-log --level info --message 'Starting deployment'",
      "/communicate-log --level error --message 'Build failed: {reason}'"
    ],
    "available_to": "ALL agents"
  },

  "agent-delegate": {
    "command": "/agent-delegate --to {specialist} --task {task} --context {context}",
    "when": "Need specialist expertise outside my domain",
    "usage": "Delegate to appropriate specialist agent",
    "examples": [
      "/agent-delegate --to security-specialist --task 'Audit authentication' --context {...}",
      "/agent-delegate --to backend-dev --task 'Implement API' --context {...}"
    ],
    "pattern": "Always include context from memory",
    "available_to": "ALL agents"
  },

  "agent-escalate": {
    "command": "/agent-escalate --to {supervisor} --issue {issue} --severity {level}",
    "when": "Encounter blocker I cannot resolve",
    "usage": "Escalate problems to supervisor or coordinator",
    "examples": [
      "/agent-escalate --to hierarchical-coordinator --issue 'Missing API credentials' --severity high",
      "/agent-escalate --to user --issue 'Requirement clarification needed' --severity medium"
    ],
    "available_to": "ALL agents"
  },

  "agent-status": {
    "command": "/agent-status --agent {agent-id}",
    "when": "Need to check agent progress",
    "usage": "Monitor delegated tasks",
    "examples": [
      "/agent-status --agent backend-dev-123"
    ],
    "available_to": "ALL agents"
  },

  "memory-store": {
    "command": "/memory-store --key {namespace/key} --value {value}",
    "when": "Need to persist data for other agents",
    "usage": "Share results, context, or state",
    "examples": [
      "/memory-store --key 'backend-dev/api-v2/schema' --value {...}",
      "/memory-store --key 'marketing/campaign-123/results' --value {...}"
    ],
    "pattern": "Namespace: {agent-role}/{task-id}/{data-type}",
    "available_to": "ALL agents"
  },

  "memory-retrieve": {
    "command": "/memory-retrieve --key {namespace/key}",
    "when": "Need to load previously stored context",
    "usage": "Access shared data from other agents",
    "examples": [
      "/memory-retrieve --key 'backend-dev/api-v2/schema'",
      "/memory-retrieve --key 'marketing/campaign-123/audience-data'"
    ],
    "available_to": "ALL agents"
  }
}
```

### Testing & Validation (6 commands)

```javascript
// Store in memory namespace: universal-commands/testing/*

{
  "test-run": {
    "command": "/test-run --suite {suite}",
    "when": "Need to execute tests",
    "usage": "Run test suites to validate functionality",
    "examples": [
      "/test-run --suite unit",
      "/test-run --suite integration",
      "/test-run --suite e2e"
    ],
    "available_to": "ALL agents"
  },

  "test-coverage": {
    "command": "/test-coverage --threshold {percent}",
    "when": "Need to check code coverage",
    "usage": "Verify test coverage meets requirements",
    "examples": [
      "/test-coverage --threshold 80",
      "/test-coverage"
    ],
    "available_to": "ALL agents"
  },

  "test-validate": {
    "command": "/test-validate --target {target}",
    "when": "Need to validate outputs or behavior",
    "usage": "Verify results meet specifications",
    "examples": [
      "/test-validate --target api-endpoints",
      "/test-validate --target data-integrity"
    ],
    "available_to": "ALL agents"
  },

  "lint-check": {
    "command": "/lint-check --files {pattern}",
    "when": "Need to check code style",
    "usage": "Ensure code follows style guidelines",
    "examples": [
      "/lint-check --files 'src/**/*.js'",
      "/lint-check"
    ],
    "available_to": "ALL agents"
  },

  "format-check": {
    "command": "/format-check --files {pattern}",
    "when": "Need to verify code formatting",
    "usage": "Check consistent code formatting",
    "examples": [
      "/format-check --files 'src/**/*.ts'"
    ],
    "available_to": "ALL agents"
  },

  "type-check": {
    "command": "/type-check",
    "when": "Need to verify TypeScript types",
    "usage": "Validate type safety",
    "examples": [
      "/type-check"
    ],
    "available_to": "ALL agents"
  }
}
```

### Utilities (10 commands)

```javascript
// Store in memory namespace: universal-commands/utilities/*

{
  "json-validator": {
    "command": "/json-validator --file {file}",
    "when": "Need to validate JSON syntax",
    "usage": "Verify JSON is well-formed",
    "examples": [
      "/json-validator --file config.json"
    ],
    "available_to": "ALL agents"
  },

  "yaml-parser": {
    "command": "/yaml-parser --file {file}",
    "when": "Need to parse YAML files",
    "usage": "Read and validate YAML configuration",
    "examples": [
      "/yaml-parser --file .github/workflows/ci.yml"
    ],
    "available_to": "ALL agents"
  },

  "markdown-gen": {
    "command": "/markdown-gen --template {template} --data {data}",
    "when": "Need to generate markdown documentation",
    "usage": "Create formatted documentation",
    "examples": [
      "/markdown-gen --template api-docs --data {...}"
    ],
    "available_to": "ALL agents"
  },

  "regex-helper": {
    "command": "/regex-helper --pattern {pattern} --test {string}",
    "when": "Need regex pattern assistance",
    "usage": "Test and validate regex patterns",
    "examples": [
      "/regex-helper --pattern '^[a-z]+$' --test 'hello'"
    ],
    "available_to": "ALL agents"
  },

  "data-transform": {
    "command": "/data-transform --from {format} --to {format} --data {data}",
    "when": "Need to convert data formats",
    "usage": "Transform between JSON, YAML, CSV, etc.",
    "examples": [
      "/data-transform --from json --to yaml --data {...}"
    ],
    "available_to": "ALL agents"
  },

  "api-mock": {
    "command": "/api-mock --endpoint {endpoint} --response {response}",
    "when": "Need to mock API responses",
    "usage": "Create mock endpoints for testing",
    "examples": [
      "/api-mock --endpoint '/api/users' --response {...}"
    ],
    "available_to": "ALL agents"
  },

  "csv-processor": {
    "command": "/csv-processor --file {file} --operation {op}",
    "when": "Need to process CSV data",
    "usage": "Parse, validate, transform CSV files",
    "examples": [
      "/csv-processor --file data.csv --operation parse"
    ],
    "available_to": "ALL agents"
  },

  "context-save": {
    "command": "/context-save --name {name}",
    "when": "Need to save current context",
    "usage": "Preserve working state for later",
    "examples": [
      "/context-save --name feature-implementation-state"
    ],
    "available_to": "ALL agents"
  },

  "context-restore": {
    "command": "/context-restore --name {name}",
    "when": "Need to restore previous context",
    "usage": "Resume from saved state",
    "examples": [
      "/context-restore --name feature-implementation-state"
    ],
    "available_to": "ALL agents"
  },

  "batch-operation": {
    "command": "/batch-operation --commands {commands}",
    "when": "Need to execute multiple commands",
    "usage": "Run batch of related commands",
    "examples": [
      "/batch-operation --commands ['test-run', 'lint-check', 'format-check']"
    ],
    "available_to": "ALL agents"
  }
}
```

---

## Universal MCP Tools Storage

### Coordination Tools (10 tools)

```javascript
// Store in memory namespace: universal-tools/mcp/coordination/*

{
  "swarm_init": {
    "tool": "mcp__ruv-swarm__swarm_init",
    "when": "Need to initialize agent coordination",
    "usage": "Set up swarm topology for multi-agent tasks",
    "params": {
      "topology": "mesh | hierarchical | ring | star",
      "maxAgents": "1-100",
      "strategy": "balanced | specialized | adaptive"
    },
    "examples": [
      "mcp__ruv-swarm__swarm_init({ topology: 'mesh', maxAgents: 8, strategy: 'adaptive' })"
    ],
    "available_to": "ALL agents"
  },

  "agent_spawn": {
    "tool": "mcp__ruv-swarm__agent_spawn",
    "when": "Need to spawn specialist sub-agents",
    "usage": "Create agents for specific tasks",
    "params": {
      "type": "researcher | coder | analyst | optimizer | coordinator",
      "capabilities": "array of capabilities",
      "name": "descriptive identifier"
    },
    "examples": [
      "mcp__ruv-swarm__agent_spawn({ type: 'coder', name: 'backend-api-dev', capabilities: ['nodejs', 'express'] })"
    ],
    "available_to": "ALL agents"
  },

  "task_orchestrate": {
    "tool": "mcp__ruv-swarm__task_orchestrate",
    "when": "Need to coordinate complex tasks",
    "usage": "Distribute work across swarm",
    "params": {
      "task": "task description",
      "strategy": "parallel | sequential | adaptive",
      "priority": "low | medium | high | critical",
      "maxAgents": "1-10"
    },
    "examples": [
      "mcp__ruv-swarm__task_orchestrate({ task: 'Build REST API', strategy: 'adaptive', priority: 'high' })"
    ],
    "available_to": "ALL agents"
  },

  "task_status": {
    "tool": "mcp__ruv-swarm__task_status",
    "when": "Need to check task progress",
    "usage": "Monitor orchestrated tasks",
    "available_to": "ALL agents"
  },

  "task_results": {
    "tool": "mcp__ruv-swarm__task_results",
    "when": "Need to retrieve task outputs",
    "usage": "Get completed task results",
    "available_to": "ALL agents"
  },

  "agent_list": {
    "tool": "mcp__ruv-swarm__agent_list",
    "when": "Need to see active agents",
    "usage": "List all spawned agents and their status",
    "available_to": "ALL agents"
  },

  "agent_metrics": {
    "tool": "mcp__ruv-swarm__agent_metrics",
    "when": "Need agent performance data",
    "usage": "Monitor agent efficiency and resource usage",
    "available_to": "ALL agents"
  },

  "swarm_status": {
    "tool": "mcp__ruv-swarm__swarm_status",
    "when": "Need swarm health check",
    "usage": "Monitor overall swarm state",
    "available_to": "ALL agents"
  },

  "swarm_monitor": {
    "tool": "mcp__ruv-swarm__swarm_monitor",
    "when": "Need real-time monitoring",
    "usage": "Watch swarm activity live",
    "available_to": "ALL agents"
  },

  "memory_usage": {
    "tool": "mcp__ruv-swarm__memory_usage",
    "when": "Need memory statistics",
    "usage": "Check memory consumption across swarm",
    "available_to": "ALL agents"
  }
}
```

### Memory Tools (3 tools)

```javascript
// Store in memory namespace: universal-tools/mcp/memory/*

{
  "memory_store": {
    "tool": "mcp__claude-flow__memory_store",
    "when": "Need to persist data for agents",
    "usage": "Store cross-agent shared data",
    "namespace_pattern": "{agent-role}/{task-id}/{data-type}",
    "examples": [
      "mcp__claude-flow__memory_store({ key: 'backend-dev/api-v2/schema', value: {...}, ttl: 86400 })"
    ],
    "available_to": "ALL agents"
  },

  "memory_retrieve": {
    "tool": "mcp__claude-flow__memory_retrieve",
    "when": "Need to load stored context",
    "usage": "Access data from other agents",
    "available_to": "ALL agents"
  },

  "memory_search": {
    "tool": "mcp__claude-flow__memory_search",
    "when": "Need to find data by pattern",
    "usage": "Search across memory namespaces",
    "available_to": "ALL agents"
  }
}
```

---

## Implementation: Store in MCP Memory

### Script to Store All Universal Data

```javascript
// Store all universal commands and tools in memory
async function storeUniversalData() {
  const categories = [
    'file-operations',
    'git-operations',
    'communication',
    'testing',
    'utilities'
  ];

  for (const category of categories) {
    const commands = await loadCommands(category);

    for (const [name, config] of Object.entries(commands)) {
      await mcp__claude-flow__memory_store({
        key: `universal-commands/${category}/${name}`,
        value: config,
        ttl: null  // No expiration for universal data
      });
    }
  }

  // Store MCP tools
  const mcp_categories = ['coordination', 'memory'];

  for (const category of mcp_categories) {
    const tools = await loadTools(category);

    for (const [name, config] of Object.entries(tools)) {
      await mcp__claude-flow__memory_store({
        key: `universal-tools/mcp/${category}/${name}`,
        value: config,
        ttl: null
      });
    }
  }

  console.log('✅ Universal commands and tools stored in memory');
}
```

### Agent Access Pattern

```javascript
// Each agent can retrieve universal data
async function getUniversalCommands(category) {
  return await mcp__claude-flow__memory_search({
    pattern: `universal-commands/${category}/*`
  });
}

async function getUniversalTools(category) {
  return await mcp__claude-flow__memory_search({
    pattern: `universal-tools/mcp/${category}/*`
  });
}

// Example: Get all file operations
const fileOps = await getUniversalCommands('file-operations');
// Returns: { 'file-read': {...}, 'file-write': {...}, ... }
```

---

## Next Steps

1. ✅ Document memory storage patterns (this file)
2. ⏳ Execute storage script to populate memory
3. ⏳ Validate memory retrieval works
4. ⏳ Update agent templates to reference memory

---

**Status**: Memory patterns documented
**Ready for**: Execution and agent integration
