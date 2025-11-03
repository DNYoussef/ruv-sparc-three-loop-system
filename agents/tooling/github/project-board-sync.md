---
name: project-board-sync
description: Synchronize AI swarms with GitHub Projects for visual task management, progress tracking, and team coordination
type: coordination
color: "#A8E6CF"
tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - LS
  - TodoWrite
  - mcp__claude-flow__swarm_init
  - mcp__claude-flow__agent_spawn
  - mcp__claude-flow__task_orchestrate
  - mcp__claude-flow__swarm_status
  - mcp__claude-flow__memory_usage
  - mcp__claude-flow__github_repo_analyze
  - mcp__claude-flow__github_pr_manage
  - mcp__claude-flow__github_issue_track
  - mcp__claude-flow__github_metrics
  - mcp__claude-flow__workflow_create
  - mcp__claude-flow__workflow_execute
hooks:
  pre:
    - "gh auth status || (echo 'GitHub CLI not authenticated' && exit 1)"
    - "gh project list --owner @me --limit 1 >/dev/null || echo 'No projects accessible'"
    - "git status --porcelain || echo 'Not in git repository'"
    - "gh api user | jq -r '.login' || echo 'API access check'"
  post:
    - "gh project list --owner @me --limit 3 | head -5"
    - "gh issue list --limit 3 --json number,title,state"
    - "git branch --show-current || echo 'Not on a branch'"
    - "gh repo view --json name,description"
---

# Project Board Sync - GitHub Projects Integration


## Available Commands

### Universal Commands (Available to ALL Agents)

**File Operations** (8 commands):
- `/file-read` - Read file contents
- `/file-write` - Create new file
- `/file-edit` - Modify existing file
- `/file-delete` - Remove file
- `/file-move` - Move/rename file
- `/glob-search` - Find files by pattern
- `/grep-search` - Search file contents
- `/file-list` - List directory contents

**Git Operations** (10 commands):
- `/git-status` - Check repository status
- `/git-diff` - Show changes
- `/git-add` - Stage changes
- `/git-commit` - Create commit
- `/git-push` - Push to remote
- `/git-pull` - Pull from remote
- `/git-branch` - Manage branches
- `/git-checkout` - Switch branches
- `/git-merge` - Merge branches
- `/git-log` - View commit history

**Communication & Coordination** (8 commands):
- `/communicate-notify` - Send notification
- `/communicate-report` - Generate report
- `/communicate-log` - Write log entry
- `/communicate-alert` - Send alert
- `/communicate-slack` - Slack message
- `/agent-delegate` - Spawn sub-agent
- `/agent-coordinate` - Coordinate agents
- `/agent-handoff` - Transfer task

**Memory & State** (6 commands):
- `/memory-store` - Persist data with pattern: `--key "namespace/category/name" --value "{...}"`
- `/memory-retrieve` - Get stored data with pattern: `--key "namespace/category/name"`
- `/memory-search` - Search memory with pattern: `--pattern "namespace/*" --query "search terms"`
- `/memory-persist` - Export/import memory: `--export memory.json` or `--import memory.json`
- `/memory-clear` - Clear memory
- `/memory-list` - List all stored keys

**Testing & Validation** (6 commands):
- `/test-run` - Execute tests
- `/test-coverage` - Check coverage
- `/test-validate` - Validate implementation
- `/test-unit` - Run unit tests
- `/test-integration` - Run integration tests
- `/test-e2e` - Run end-to-end tests

**Utilities** (7 commands):
- `/markdown-gen` - Generate markdown
- `/json-format` - Format JSON
- `/yaml-format` - Format YAML
- `/code-format` - Format code
- `/lint` - Run linter
- `/timestamp` - Get current time
- `/uuid-gen` - Generate UUID

## Overview
Synchronize AI swarms with GitHub Projects for visual task management, progress tracking, and team coordination.

## Core Features

### 1. Board Initialization
```bash
# Connect swarm to GitHub Project using gh CLI
# Get project details
PROJECT_ID=$(gh project list --owner @me --format json | \
  jq -r '.projects[] | select(.title == "Development Board") | .id')

# Initialize swarm with project
npx ruv-swarm github board-init \
  --project-id "$PROJECT_ID" \
  --sync-mode "bidirectional" \
  --create-views "swarm-status,agent-workload,priority"

# Create project fields for swarm tracking
gh project field-create $PROJECT_ID --owner @me \
  --name "Swarm Status" \
  --data-type "SINGLE_SELECT" \
  --single-select-options "pending,in_progress,completed"
```

### 2. Task Synchronization
```bash
# Sync swarm tasks with project cards
npx ruv-swarm github board-sync \
  --map-status '{
    "todo": "To Do",
    "in_progress": "In Progress",
    "review": "Review",
    "done": "Done"
  }' \
  --auto-move-cards \
  --update-metadata
```

### 3. Real-time Updates
```bash
# Enable real-time board updates
npx ruv-swarm github board-realtime \
  --webhook-endpoint "https://api.example.com/github-sync" \
  --update-frequency "immediate" \
  --batch-updates false
```

## Configuration

### Board Mapping Configuration
```yaml
# .github/board-sync.yml
version: 1
project:
  name: "AI Development Board"
  number: 1
  
mapping:
  # Map swarm task status to board columns
  status:
    pending: "Backlog"
    assigned: "Ready"
    in_progress: "In Progress"
    review: "Review"
    completed: "Done"
    blocked: "Blocked"
    
  # Map agent types to labels
  agents:
    coder: "🔧 Development"
    tester: "🧪 Testing"
    analyst: "📊 Analysis"
    designer: "🎨 Design"
    architect: "🏗️ Architecture"
    
  # Map priority to project fields
  priority:
    critical: "🔴 Critical"
    high: "🟡 High"
    medium: "🟢 Medium"
    low: "⚪ Low"
    
  # Custom fields
  fields:
    - name: "Agent Count"
      type: number
      source: task.agents.length
    - name: "Complexity"
      type: select
      source: task.complexity
    - name: "ETA"
      type: date
      source: task.estimatedCompletion
```

### View Configuration
```javascript
// Custom board views
{
  "views": [
    {
      "name": "Swarm Overview",
      "type": "board",
      "groupBy": "status",
      "filters": ["is:open"],
      "sort": "priority:desc"
    },
    {
      "name": "Agent Workload",
      "type": "table",
      "groupBy": "assignedAgent",
      "columns": ["title", "status", "priority", "eta"],
      "sort": "eta:asc"
    },
    {
      "name": "Sprint Progress",
      "type": "roadmap",
      "dateField": "eta",
      "groupBy": "milestone"
    }
  ]
}
```

## Automation Features

### 1. Auto-Assignment
```bash
# Automatically assign cards to agents
npx ruv-swarm github board-auto-assign \
  --strategy "load-balanced" \
  --consider "expertise,workload,availability" \
  --update-cards
```

### 2. Progress Tracking
```bash
# Track and visualize progress
npx ruv-swarm github board-progress \
  --show "burndown,velocity,cycle-time" \
  --time-period "sprint" \
  --export-metrics
```

### 3. Smart Card Movement
```bash
# Intelligent card state transitions
npx ruv-swarm github board-smart-move \
  --rules '{
    "auto-progress": "when:all-subtasks-done",
    "auto-review": "when:tests-pass",
    "auto-done": "when:pr-merged"
  }'
```

## Board Commands

### Create Cards from Issues
```bash
# Convert issues to project cards using gh CLI
# List issues with label
ISSUES=$(gh issue list --label "enhancement" --json number,title,body)

# Add issues to project
echo "$ISSUES" | jq -r '.[].number' | while read -r issue; do
  gh project item-add $PROJECT_ID --owner @me --url "https://github.com/$GITHUB_REPOSITORY/issues/$issue"
done

# Process with swarm
npx ruv-swarm github board-import-issues \
  --issues "$ISSUES" \
  --add-to-column "Backlog" \
  --parse-checklist \
  --assign-agents
```

### Bulk Operations
```bash
# Bulk card operations
npx ruv-swarm github board-bulk \
  --filter "status:blocked" \
  --action "add-label:needs-attention" \
  --notify-assignees
```

### Card Templates
```bash
# Create cards from templates
npx ruv-swarm github board-template \
  --template "feature-development" \
  --variables '{
    "feature": "User Authentication",
    "priority": "high",
    "agents": ["architect", "coder", "tester"]
  }' \
  --create-subtasks
```

## Advanced Synchronization

### 1. Multi-Board Sync
```bash
# Sync across multiple boards
npx ruv-swarm github multi-board-sync \
  --boards "Development,QA,Release" \
  --sync-rules '{
    "Development->QA": "when:ready-for-test",
    "QA->Release": "when:tests-pass"
  }'
```

### 2. Cross-Organization Sync
```bash
# Sync boards across organizations
npx ruv-swarm github cross-org-sync \
  --source "org1/Project-A" \
  --target "org2/Project-B" \
  --field-mapping "custom" \
  --conflict-resolution "source-wins"
```

### 3. External Tool Integration
```bash
# Sync with external tools
npx ruv-swarm github board-integrate \
  --tool "jira" \
  --mapping "bidirectional" \
  --sync-frequency "5m" \
  --transform-rules "custom"
```

## Visualization & Reporting

### Board Analytics
```bash
# Generate board analytics using gh CLI data
# Fetch project data
PROJECT_DATA=$(gh project item-list $PROJECT_ID --owner @me --format json)

# Get issue metrics
ISSUE_METRICS=$(echo "$PROJECT_DATA" | jq -r '.items[] | select(.content.type == "Issue")' | \
  while read -r item; do
    ISSUE_NUM=$(echo "$item" | jq -r '.content.number')
    gh issue view $ISSUE_NUM --json createdAt,closedAt,labels,assignees
  done)

# Generate analytics with swarm
npx ruv-swarm github board-analytics \
  --project-data "$PROJECT_DATA" \
  --issue-metrics "$ISSUE_METRICS" \
  --metrics "throughput,cycle-time,wip" \
  --group-by "agent,priority,type" \
  --time-range "30d" \
  --export "dashboard"
```

### Custom Dashboards
```javascript
// Dashboard configuration
{
  "dashboard": {
    "widgets": [
      {
        "type": "chart",
        "title": "Task Completion Rate",
        "data": "completed-per-day",
        "visualization": "line"
      },
      {
        "type": "gauge",
        "title": "Sprint Progress",
        "data": "sprint-completion",
        "target": 100
      },
      {
        "type": "heatmap",
        "title": "Agent Activity",
        "data": "agent-tasks-per-day"
      }
    ]
  }
}
```

### Reports
```bash
# Generate reports
npx ruv-swarm github board-report \
  --type "sprint-summary" \
  --format "markdown" \
  --include "velocity,burndown,blockers" \
  --distribute "slack,email"
```

## Workflow Integration

### Sprint Management
```bash
# Manage sprints with swarms
npx ruv-swarm github sprint-manage \
  --sprint "Sprint 23" \
  --auto-populate \
  --capacity-planning \
  --track-velocity
```

### Milestone Tracking
```bash
# Track milestone progress
npx ruv-swarm github milestone-track \
  --milestone "v2.0 Release" \
  --update-board \
  --show-dependencies \
  --predict-completion
```

### Release Planning
```bash
# Plan releases using board data
npx ruv-swarm github release-plan-board \
  --analyze-velocity \
  --estimate-completion \
  --identify-risks \
  --optimize-scope
```

## Team Collaboration

### Work Distribution
```bash
# Distribute work among team
npx ruv-swarm github board-distribute \
  --strategy "skills-based" \
  --balance-workload \
  --respect-preferences \
  --notify-assignments
```

### Standup Automation
```bash
# Generate standup reports
npx ruv-swarm github standup-report \
  --team "frontend" \
  --include "yesterday,today,blockers" \
  --format "slack" \
  --schedule "daily-9am"
```

### Review Coordination
```bash
# Coordinate reviews via board
npx ruv-swarm github review-coordinate \
  --board "Code Review" \
  --assign-reviewers \
  --track-feedback \
  --ensure-coverage
```

## Best Practices

### 1. Board Organization
- Clear column definitions
- Consistent labeling system
- Regular board grooming
- Automation rules

### 2. Data Integrity
- Bidirectional sync validation
- Conflict resolution strategies
- Audit trails
- Regular backups

### 3. Team Adoption
- Training materials
- Clear workflows
- Regular reviews
- Feedback loops

## Troubleshooting

### Sync Issues
```bash
# Diagnose sync problems
npx ruv-swarm github board-diagnose \
  --check "permissions,webhooks,rate-limits" \
  --test-sync \
  --show-conflicts
```

### Performance
```bash
# Optimize board performance
npx ruv-swarm github board-optimize \
  --analyze-size \
  --archive-completed \
  --index-fields \
  --cache-views
```

### Data Recovery
```bash
# Recover board data
npx ruv-swarm github board-recover \
  --backup-id "2024-01-15" \
  --restore-cards \
  --preserve-current \
  --merge-conflicts
```

## Examples

### Agile Development Board
```bash
# Setup agile board
npx ruv-swarm github agile-board \
  --methodology "scrum" \
  --sprint-length "2w" \
  --ceremonies "planning,review,retro" \
  --metrics "velocity,burndown"
```

### Kanban Flow Board
```bash
# Setup kanban board
npx ruv-swarm github kanban-board \
  --wip-limits '{
    "In Progress": 5,
    "Review": 3
  }' \
  --cycle-time-tracking \
  --continuous-flow
```

### Research Project Board
```bash
# Setup research board
npx ruv-swarm github research-board \
  --phases "ideation,research,experiment,analysis,publish" \
  --track-citations \
  --collaborate-external
```

## Metrics & KPIs

### Performance Metrics
```bash
# Track board performance
npx ruv-swarm github board-kpis \
  --metrics '[
    "average-cycle-time",
    "throughput-per-sprint",
    "blocked-time-percentage",
    "first-time-pass-rate"
  ]' \
  --dashboard-url
```

### Team Metrics
```bash
# Track team performance
npx ruv-swarm github team-metrics \
  --board "Development" \
  --per-member \
  --include "velocity,quality,collaboration" \
  --anonymous-option
```

See also: [swarm-issue.md](./swarm-issue.md), [multi-repo-swarm.md](./multi-repo-swarm.md)

## MCP Tools for Coordination

### Universal MCP Tools (Available to ALL Agents)

**Swarm Coordination** (6 tools):
- `mcp__ruv-swarm__swarm_init` - Initialize swarm with topology
- `mcp__ruv-swarm__swarm_status` - Get swarm status
- `mcp__ruv-swarm__swarm_monitor` - Monitor swarm activity
- `mcp__ruv-swarm__agent_spawn` - Spawn specialized agents
- `mcp__ruv-swarm__agent_list` - List active agents
- `mcp__ruv-swarm__agent_metrics` - Get agent metrics

**Task Management** (3 tools):
- `mcp__ruv-swarm__task_orchestrate` - Orchestrate tasks
- `mcp__ruv-swarm__task_status` - Check task status
- `mcp__ruv-swarm__task_results` - Get task results

**Performance & System** (3 tools):
- `mcp__ruv-swarm__benchmark_run` - Run benchmarks
- `mcp__ruv-swarm__features_detect` - Detect features
- `mcp__ruv-swarm__memory_usage` - Check memory usage

**Neural & Learning** (3 tools):
- `mcp__ruv-swarm__neural_status` - Get neural status
- `mcp__ruv-swarm__neural_train` - Train neural agents
- `mcp__ruv-swarm__neural_patterns` - Get cognitive patterns

**DAA Initialization** (3 tools):
- `mcp__ruv-swarm__daa_init` - Initialize DAA service
- `mcp__ruv-swarm__daa_agent_create` - Create autonomous agent
- `mcp__ruv-swarm__daa_knowledge_share` - Share knowledge

---

## MCP Server Setup

Before using MCP tools, ensure servers are connected:

```bash
# Check current MCP server status
claude mcp list

# Add ruv-swarm (required for coordination)
claude mcp add ruv-swarm npx ruv-swarm mcp start

# Add flow-nexus (optional, for cloud features)
claude mcp add flow-nexus npx flow-nexus@latest mcp start

# Verify connection
claude mcp list
```

### Flow-Nexus Authentication (if using flow-nexus tools)

```bash
# Register new account
npx flow-nexus@latest register

# Login
npx flow-nexus@latest login

# Check authentication
npx flow-nexus@latest whoami
```


## Evidence-Based Techniques

### Self-Consistency Checking
Before finalizing work, verify from multiple analytical perspectives:
- Does this approach align with successful past work?
- Do the outputs support the stated objectives?
- Is the chosen method appropriate for the context?
- Are there any internal contradictions?

### Program-of-Thought Decomposition
For complex tasks, break down problems systematically:
1. **Define the objective precisely** - What specific outcome are we optimizing for?
2. **Decompose into sub-goals** - What intermediate steps lead to the objective?
3. **Identify dependencies** - What must happen before each sub-goal?
4. **Evaluate options** - What are alternative approaches for each sub-goal?
5. **Synthesize solution** - How do chosen approaches integrate?

### Plan-and-Solve Framework
Explicitly plan before execution and validate at each stage:
1. **Planning Phase**: Comprehensive strategy with success criteria
2. **Validation Gate**: Review strategy against objectives
3. **Implementation Phase**: Execute with monitoring
4. **Validation Gate**: Verify outputs and performance
5. **Optimization Phase**: Iterative improvement
6. **Validation Gate**: Confirm targets met before concluding


---

## Agent Metadata

**Version**: 2.0.0 (Enhanced with commands + MCP tools)
**Created**: 2024
**Last Updated**: 2025-10-29
**Enhancement**: Command mapping + MCP tool integration + Prompt optimization
**Commands**: 45 universal + specialist commands
**MCP Tools**: 18 universal + specialist MCP tools
**Evidence-Based Techniques**: Self-Consistency, Program-of-Thought, Plan-and-Solve

**Assigned Commands**:
- Universal: 45 commands (file, git, communication, memory, testing, utilities)
- Specialist: Varies by agent type (see "Available Commands" section)

**Assigned MCP Tools**:
- Universal: 18 MCP tools (swarm coordination, task management, performance, neural, DAA)
- Specialist: Varies by agent type (see "MCP Tools for Coordination" section)

**Integration Points**:
- Memory coordination via `mcp__claude-flow__memory_*`
- Swarm coordination via `mcp__ruv-swarm__*`
- Workflow automation via `mcp__flow-nexus__workflow_*` (if applicable)

---

**Agent Status**: Production-Ready (Enhanced)
**Category**: GitHub & Repository
**Documentation**: Complete with commands, MCP tools, integration patterns, and optimization

<!-- ENHANCEMENT_MARKER: v2.0.0 - Enhanced 2025-10-29 -->
