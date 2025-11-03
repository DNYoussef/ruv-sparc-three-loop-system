---
name: github-modes
description: Comprehensive GitHub integration modes for workflow orchestration, PR management, and repository coordination with batch optimization
tools: mcp__claude-flow__swarm_init, mcp__claude-flow__agent_spawn, mcp__claude-flow__task_orchestrate, Bash, TodoWrite, Read, Write
color: purple
type: development
capabilities:
  - GitHub workflow orchestration
  - Pull request management and review
  - Issue tracking and coordination
  - Release management and deployment
  - Repository architecture and organization
  - CI/CD pipeline coordination
priority: medium
hooks:
  pre: |
    echo "Starting github-modes..."
    echo "Initializing GitHub workflow coordination"
    gh auth status || (echo "GitHub CLI authentication required" && exit 1)
    git status > /dev/null || (echo "Not in a git repository" && exit 1)
  post: |
    echo "Completed github-modes"
    echo "GitHub operations synchronized"
    echo "Workflow coordination finalized"
---

# GitHub Integration Modes


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
This document describes all GitHub integration modes available in Claude-Flow with ruv-swarm coordination. Each mode is optimized for specific GitHub workflows and includes batch tool integration for maximum efficiency.

## GitHub Workflow Modes

### gh-coordinator
**GitHub workflow orchestration and coordination**
- **Coordination Mode**: Hierarchical
- **Max Parallel Operations**: 10
- **Batch Optimized**: Yes
- **Tools**: gh CLI commands, TodoWrite, TodoRead, Task, Memory, Bash
- **Usage**: `/github gh-coordinator <GitHub workflow description>`
- **Best For**: Complex GitHub workflows, multi-repo coordination

### pr-manager
**Pull request management and review coordination**
- **Review Mode**: Automated
- **Multi-reviewer**: Yes
- **Conflict Resolution**: Intelligent
- **Tools**: gh pr create, gh pr view, gh pr review, gh pr merge, TodoWrite, Task
- **Usage**: `/github pr-manager <PR management task>`
- **Best For**: PR reviews, merge coordination, conflict resolution

### issue-tracker
**Issue management and project coordination**
- **Issue Workflow**: Automated
- **Label Management**: Smart
- **Progress Tracking**: Real-time
- **Tools**: gh issue create, gh issue edit, gh issue comment, gh issue list, TodoWrite
- **Usage**: `/github issue-tracker <issue management task>`
- **Best For**: Project management, issue coordination, progress tracking

### release-manager
**Release coordination and deployment**
- **Release Pipeline**: Automated
- **Versioning**: Semantic
- **Deployment**: Multi-stage
- **Tools**: gh pr create, gh pr merge, gh release create, Bash, TodoWrite
- **Usage**: `/github release-manager <release task>`
- **Best For**: Release management, version coordination, deployment pipelines

## Repository Management Modes

### repo-architect
**Repository structure and organization**
- **Structure Optimization**: Yes
- **Multi-repo**: Support
- **Template Management**: Advanced
- **Tools**: gh repo create, gh repo clone, git commands, Write, Read, Bash
- **Usage**: `/github repo-architect <repository management task>`
- **Best For**: Repository setup, structure optimization, multi-repo management

### code-reviewer
**Automated code review and quality assurance**
- **Review Quality**: Deep
- **Security Analysis**: Yes
- **Performance Check**: Automated
- **Tools**: gh pr view --json files, gh pr review, gh pr comment, Read, Write
- **Usage**: `/github code-reviewer <review task>`
- **Best For**: Code quality, security reviews, performance analysis

### branch-manager
**Branch management and workflow coordination**
- **Branch Strategy**: GitFlow
- **Merge Strategy**: Intelligent
- **Conflict Prevention**: Proactive
- **Tools**: gh api (for branch operations), git commands, Bash
- **Usage**: `/github branch-manager <branch management task>`
- **Best For**: Branch coordination, merge strategies, workflow management

## Integration Commands

### sync-coordinator
**Multi-package synchronization**
- **Package Sync**: Intelligent
- **Version Alignment**: Automatic
- **Dependency Resolution**: Advanced
- **Tools**: git commands, gh pr create, Read, Write, Bash
- **Usage**: `/github sync-coordinator <sync task>`
- **Best For**: Package synchronization, version management, dependency updates

### ci-orchestrator
**CI/CD pipeline coordination**
- **Pipeline Management**: Advanced
- **Test Coordination**: Parallel
- **Deployment**: Automated
- **Tools**: gh pr checks, gh workflow list, gh run list, Bash, TodoWrite, Task
- **Usage**: `/github ci-orchestrator <CI/CD task>`
- **Best For**: CI/CD coordination, test management, deployment automation

### security-guardian
**Security and compliance management**
- **Security Scan**: Automated
- **Compliance Check**: Continuous
- **Vulnerability Management**: Proactive
- **Tools**: gh search code, gh issue create, gh secret list, Read, Write
- **Usage**: `/github security-guardian <security task>`
- **Best For**: Security audits, compliance checks, vulnerability management

## Usage Examples

### Creating a coordinated pull request workflow:
```bash
/github pr-manager "Review and merge feature/new-integration branch with automated testing and multi-reviewer coordination"
```

### Managing repository synchronization:
```bash
/github sync-coordinator "Synchronize claude-code-flow and ruv-swarm packages, align versions, and update cross-dependencies"
```

### Setting up automated issue tracking:
```bash
/github issue-tracker "Create and manage integration issues with automated progress tracking and swarm coordination"
```

## Batch Operations

All GitHub modes support batch operations for maximum efficiency:

### Parallel GitHub Operations Example:
```javascript
[Single Message with BatchTool]:
  Bash("gh issue create --title 'Feature A' --body '...'")
  Bash("gh issue create --title 'Feature B' --body '...'")
  Bash("gh pr create --title 'PR 1' --head 'feature-a' --base 'main'")
  Bash("gh pr create --title 'PR 2' --head 'feature-b' --base 'main'")
  TodoWrite { todos: [todo1, todo2, todo3] }
  Bash("git checkout main && git pull")
```

## Integration with ruv-swarm

All GitHub modes can be enhanced with ruv-swarm coordination:

```javascript
// Initialize swarm for GitHub workflow
mcp__claude-flow__swarm_init { topology: "hierarchical", maxAgents: 5 }
mcp__claude-flow__agent_spawn { type: "coordinator", name: "GitHub Coordinator" }
mcp__claude-flow__agent_spawn { type: "reviewer", name: "Code Reviewer" }
mcp__claude-flow__agent_spawn { type: "tester", name: "QA Agent" }

// Execute GitHub workflow with coordination
mcp__claude-flow__task_orchestrate { task: "GitHub workflow", strategy: "parallel" }
```

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
