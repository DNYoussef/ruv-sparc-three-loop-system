# MCP Marketplace Guide - Complete MCP Server Ecosystem

This document provides a comprehensive guide to the Model Context Protocol (MCP) server ecosystem, including official servers, community servers, and agent-to-MCP-server mappings for the ruv-SPARC Three-Loop System.

## Table of Contents

1. [Currently Installed MCP Servers](#currently-installed-mcp-servers)
2. [Official Reference Servers](#official-reference-servers)
3. [Recommended MCP Servers by Category](#recommended-mcp-servers-by-category)
4. [Agent-to-MCP-Server Mapping](#agent-to-mcp-server-mapping)
5. [Installation Guide](#installation-guide)
6. [Usage Examples](#usage-examples)

---

## Currently Installed MCP Servers

### 1. **connascence-analyzer** (Production Ready)
**Repository**: https://github.com/DNYoussef/connascence-safety-analyzer

**What It Does**:
- Analyzes code for 9 connascence types and 7+ violation categories
- Detects God Objects, Parameter Bombs, Deep Nesting, Cyclomatic Complexity
- NASA Power of 10 Rules compliance checking
- Performance: 0.018 seconds per file analysis

**Available Tools**:
- `analyze_file(file_path, analysis_type)` - Analyze single file
- `analyze_workspace(workspace_path, file_patterns)` - Batch analysis
- `health_check()` - Server status

**Agents with Access**: 14 code quality agents
- coder, reviewer, tester, code-analyzer
- functionality-audit, theater-detection-audit, production-validator
- sparc-coder, analyst, backend-dev, mobile-dev, ml-developer
- base-template-generator, code-review-swarm

---

### 2. **memory-mcp** (Production Ready)
**Repository**: https://github.com/DNYoussef/memory-mcp-triple-system

**What It Does**:
- Persistent cross-session memory with triple-layer retention (24h/7d/30d+)
- Mode-aware context adaptation (Execution/Planning/Brainstorming)
- 384-dimensional vector embeddings with HNSW indexing
- Semantic chunking and retrieval

**Available Tools**:
- `vector_search(query, limit)` - Semantic search with mode detection
- `memory_store(text, metadata)` - Store with automatic tagging

**Tagging Protocol** (REQUIRED for ALL writes):
- WHO: Agent name, category, capabilities
- WHEN: ISO/Unix/readable timestamps
- PROJECT: Auto-detected from working directory
- WHY: 8 intent categories (implementation, bugfix, refactor, testing, documentation, analysis, planning, research)

**Agents with Access**: ALL 86+ agents (global access)

---

### 3. **focused-changes**
**Location**: `C:\Users\17175\Documents\Cline\MCP\focused-changes-server`

**What It Does**:
- Tracks file changes to ensure they remain focused on task
- Analyzes proposed changes for scope creep
- Builds error tree diagrams from test failures

**Available Tools**:
- `start_tracking(filepath, content)` - Begin tracking file changes
- `analyze_changes(newContent)` - Validate focused changes
- `root_cause_analysis(testResults)` - Error tree from test logs

**Recommended Agents**:
- coder, reviewer, tester, sparc-coder, functionality-audit, production-validator
- All agents working on focused feature implementation

---

### 4. **ToC** (Table of Contents Generator)
**Location**: `C:\Users\17175\Documents\Cline\MCP\toc-server`

**What It Does**:
- Generates hierarchical table of contents for documentation
- Extracts descriptions from Python docstrings, classes, functions
- Supports Markdown headings, JSON, YAML, TXT files
- Filters out hidden files, __pycache__, node_modules, .log files

**Recommended Agents**:
- Documentation specialists
- Agents creating project overviews
- Architecture planning agents

---

## Official Reference Servers

From https://github.com/modelcontextprotocol/servers (Anthropic-maintained)

### 1. **Everything**
Reference/test server demonstrating all MCP capabilities (prompts, resources, tools)

### 2. **Fetch**
Web content fetching and conversion for efficient LLM usage

**Recommended Agents**: researcher, planner, web-research tasks

### 3. **Filesystem**
Secure file operations with configurable access controls

**Recommended Agents**: ALL agents needing file I/O beyond Claude Code's built-in tools

### 4. **Git**
Tools to read, search, and manipulate Git repositories

**Recommended Agents**: github-modes, pr-manager, release-manager, repo-architect, cicd-engineer

### 5. **Memory** (Official Anthropic Version)
Knowledge graph-based persistent memory system

**Note**: We use the enhanced Memory MCP Triple System instead (see above)

### 6. **Sequential Thinking**
Dynamic and reflective problem-solving through thought sequences

**Recommended Agents**: planner, researcher, system-architect, specification, pseudocode, architecture

### 7. **Time**
Time and timezone conversion capabilities

**Recommended Agents**: Scheduling, planning, and time-sensitive workflow agents

---

## Recommended MCP Servers by Category

### Development & Infrastructure

#### **GitHub**
Repository and workflow management
- **Agents**: github-modes, pr-manager, code-review-swarm, issue-tracker, release-manager, workflow-automation, project-board-sync, repo-architect, multi-repo-swarm

#### **Docker**
Container management and deployment
- **Agents**: cicd-engineer, backend-dev, production-validator

#### **E2B**
Run code in secure sandboxes
- **Agents**: tester, functionality-audit, production-validator

#### **AWS Suite** (CDK, Core, Cost Analysis, Documentation, Bedrock KB)
Cloud development and infrastructure tools
- **Agents**: backend-dev, cicd-engineer, system-architect

#### **Azure DevOps**
Pipeline and project management
- **Agents**: cicd-engineer, workflow-automation, project-board-sync

### Data & Databases

#### **PostgreSQL/Supabase**
Connects to Supabase platform for database, auth, edge functions
- **Agents**: backend-dev, mobile-dev, system-architect

#### **Neo4j**
Graph database with Cypher support
- **Agents**: backend-dev, data-modeling specialists

#### **DuckDB/MotherDuck**
Analytics database access
- **Agents**: ml-developer, data analysis agents

#### **Milvus/Chroma**
Vector database operations (embeddings)
- **Agents**: ml-developer, memory-coordinator, neural specialists

### Web & Data Extraction

#### **Firecrawl**
Web data extraction with HTML/markdown conversion
- **Agents**: researcher, web-research tasks

#### **Exa**
Search engine made for AIs
- **Agents**: researcher, planner

#### **Tavily**
Search engine for AI agents (search + extract)
- **Agents**: researcher, specification, planning agents

#### **Browserbase**
Cloud browser automation
- **Agents**: web automation, testing agents

### AI & Machine Learning

#### **Langfuse**
Open-source tool for collaborative prompt editing, versioning, evaluation
- **Agents**: prompt-architect, agent-creator, sparc-coord

#### **Logfire**
OpenTelemetry traces and metrics access
- **Agents**: perf-analyzer, performance-benchmarker

#### **Chroma**
Vector search and embeddings
- **Agents**: ml-developer, memory-coordinator, agentdb specialists

### Communication & Productivity

#### **Slack Integration**
Team communication
- **Agents**: issue-tracker, project-board-sync, workflow-automation

#### **Notion/Taskade**
Task management and documentation
- **Agents**: planner, project-board-sync, issue-tracker

### Specialized Services

#### **ElevenLabs**
Text-to-speech generation
- **Agents**: Content creation, accessibility agents

#### **Stripe/PayPal**
Payment processing
- **Agents**: backend-dev for e-commerce projects

#### **SonarQube**
Code quality and security analysis
- **Agents**: reviewer, code-analyzer, security-manager

---

## Agent-to-MCP-Server Mapping

### Core Development Agents

**coder** (Implementation Specialist)
- **MCP Servers**: memory-mcp (required), connascence-analyzer, focused-changes, filesystem, git
- **Usage**: Use connascence-analyzer before committing code, memory-mcp to log implementation decisions, focused-changes to stay on task

**reviewer** (Code Review Specialist)
- **MCP Servers**: memory-mcp (required), connascence-analyzer, focused-changes, git
- **Usage**: Run connascence-analyzer for quality checks, review change scope with focused-changes

**tester** (Testing Specialist)
- **MCP Servers**: memory-mcp (required), connascence-analyzer, focused-changes, e2b (if available)
- **Usage**: Use root_cause_analysis from focused-changes for test failures, store test patterns in memory-mcp

**planner** (Task Decomposition)
- **MCP Servers**: memory-mcp (required), sequential-thinking, ToC, fetch
- **Usage**: Use sequential-thinking for complex planning, retrieve prior plans from memory-mcp

**researcher** (Pattern Discovery)
- **MCP Servers**: memory-mcp (required), fetch, tavily/exa (if available), firecrawl
- **Usage**: Search web with tavily/exa, store research findings in memory-mcp with "research" intent

### Swarm Coordination

**hierarchical-coordinator**, **mesh-coordinator**, **adaptive-coordinator**
- **MCP Servers**: memory-mcp (required)
- **Usage**: Store coordination state, swarm topology decisions, performance metrics

**collective-intelligence-coordinator**, **swarm-memory-manager**
- **MCP Servers**: memory-mcp (required), chroma (if available)
- **Usage**: Centralized memory coordination across swarm agents

### GitHub & Repository Agents

**github-modes**, **pr-manager**, **code-review-swarm**, **issue-tracker**, **release-manager**, **workflow-automation**, **project-board-sync**, **repo-architect**, **multi-repo-swarm**
- **MCP Servers**: memory-mcp (required), git, github (if available), slack (if available)
- **Usage**: Track PR state, issue progress, release notes in memory-mcp

### SPARC Methodology

**sparc-coord**, **specification**, **pseudocode**, **architecture**, **refinement**, **sparc-coder**
- **MCP Servers**: memory-mcp (required), connascence-analyzer (for sparc-coder), sequential-thinking, ToC
- **Usage**: Store specifications, pseudocode, architecture decisions in memory-mcp with appropriate layer (mid-term/long-term)

### Performance & Optimization

**perf-analyzer**, **performance-benchmarker**, **task-orchestrator**
- **MCP Servers**: memory-mcp (required), logfire (if available)
- **Usage**: Store performance metrics, bottleneck analysis in memory-mcp

### Specialized Development

**backend-dev**
- **MCP Servers**: memory-mcp (required), connascence-analyzer, supabase/postgres (if available), aws/azure (if available)
- **Usage**: Database schema in memory-mcp, API contracts, infrastructure decisions

**mobile-dev**
- **MCP Servers**: memory-mcp (required), connascence-analyzer, supabase (if available)
- **Usage**: Platform-specific patterns, UI components in memory-mcp

**ml-developer**
- **MCP Servers**: memory-mcp (required), connascence-analyzer, chroma/milvus (if available), duckdb (if available)
- **Usage**: Model architectures, training hyperparameters, dataset metadata in memory-mcp

**cicd-engineer**
- **MCP Servers**: memory-mcp (required), git, github (if available), docker (if available), aws/azure (if available)
- **Usage**: Pipeline configurations, deployment histories in memory-mcp

**api-docs**
- **MCP Servers**: memory-mcp (required), ToC, fetch
- **Usage**: Generate API documentation, store OpenAPI schemas in memory-mcp

---

## Installation Guide

### Installing Official MCP Servers

#### 1. Fetch Server (Web Content)
```bash
npx @modelcontextprotocol/server-fetch
```

#### 2. Git Server
```bash
npx @modelcontextprotocol/server-git
```

#### 3. Sequential Thinking
```bash
npx @modelcontextprotocol/server-sequential-thinking
```

#### 4. Filesystem (with access controls)
```bash
npx @modelcontextprotocol/server-filesystem /allowed/path
```

### Installing Community MCP Servers

#### GitHub Server
```bash
npx @modelcontextprotocol/server-github
```

#### Supabase
```bash
npx supabase-mcp
```

#### Tavily (AI Search)
```bash
npx tavily-mcp
# Requires TAVILY_API_KEY environment variable
```

#### E2B (Code Sandboxes)
```bash
npx @e2b/mcp-server
# Requires E2B_API_KEY environment variable
```

### Configuring in Claude Code

Add to `C:\Users\17175\AppData\Roaming\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fetch": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-fetch"]
    },
    "git": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-git"]
    },
    "github": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your-github-token"
      }
    },
    "tavily": {
      "command": "npx",
      "args": ["tavily-mcp"],
      "env": {
        "TAVILY_API_KEY": "your-tavily-key"
      }
    }
  }
}
```

---

## Usage Examples

### Example 1: Code Quality Agent with Connascence Analyzer

```javascript
// coder agent workflow
const { analyzedFile } = await mcp.connascence_analyzer.analyze_file({
  file_path: 'src/api/users.js',
  analysis_type: 'full'
});

// Store violations in memory
await mcp.memory_mcp.memory_store({
  text: `Connascence violations found: ${JSON.stringify(analyzedFile.violations)}`,
  metadata: {
    agent: 'coder',
    project: 'api-service',
    intent: 'analysis',
    layer: 'mid_term'
  }
});

// Fix violations
// ... implementation code ...

// Track changes stayed focused
await mcp.focused_changes.analyze_changes({
  newContent: updatedFileContent
});
```

### Example 2: Research Agent with Web Search

```javascript
// researcher agent workflow
const searchResults = await mcp.tavily.search({
  query: 'best practices for GraphQL authentication 2025',
  max_results: 10
});

// Store research in memory with tagging
await mcp.memory_mcp.memory_store({
  text: `GraphQL Auth Research: ${JSON.stringify(searchResults)}`,
  metadata: {
    agent: 'researcher',
    project: 'graphql-api',
    intent: 'research',
    layer: 'long_term',
    keywords: ['graphql', 'authentication', 'best-practices']
  }
});
```

### Example 3: GitHub Agent with PR Management

```javascript
// pr-manager agent workflow
const prData = await mcp.github.get_pr({
  repo: 'owner/repo',
  pr_number: 123
});

// Analyze code changes
const analysis = await mcp.connascence_analyzer.analyze_workspace({
  workspace_path: prData.changed_files
});

// Store PR review in memory
await mcp.memory_mcp.memory_store({
  text: `PR #123 Review: ${analysis.summary.total_violations} violations found`,
  metadata: {
    agent: 'pr-manager',
    project: 'repo-name',
    intent: 'analysis',
    pr_id: 123,
    layer: 'mid_term'
  }
});
```

### Example 4: SPARC Workflow with Sequential Thinking

```javascript
// specification agent workflow
const thoughtSequence = await mcp.sequential_thinking.process({
  problem: 'Design authentication system for multi-tenant SaaS',
  context: 'Enterprise security requirements, OAuth2, JWT, role-based access'
});

// Store specification in memory (long-term)
await mcp.memory_mcp.memory_store({
  text: `Auth System Specification:\n${thoughtSequence.conclusion}`,
  metadata: {
    agent: 'specification',
    project: 'saas-platform',
    intent: 'planning',
    layer: 'long_term',
    sparc_phase: 'specification'
  }
});
```

---

## Best Practices

### Memory MCP Tagging Protocol

**ALWAYS include these metadata fields**:
- `agent`: Agent name from registry
- `project`: Auto-detected or specified
- `intent`: One of 8 categories (implementation, bugfix, refactor, testing, documentation, analysis, planning, research)
- `layer`: short_term (24h), mid_term (7d), long_term (30d+)

**Optional but recommended**:
- `task_id`: Link to specific task/issue
- `keywords`: Array of searchable keywords
- `sparc_phase`: For SPARC agents (specification, pseudocode, architecture, refinement, code)
- `pr_id`, `issue_id`, `commit_sha`: For GitHub agents

### MCP Server Selection by Task Type

| Task Type | Primary MCP Server | Secondary MCP Servers |
|-----------|-------------------|----------------------|
| Code Implementation | connascence-analyzer | memory-mcp, focused-changes, git |
| Code Review | connascence-analyzer | memory-mcp, focused-changes, git, github |
| Testing | focused-changes (root_cause) | memory-mcp, e2b |
| Research | tavily/exa | memory-mcp, fetch, firecrawl |
| Planning | sequential-thinking | memory-mcp, ToC |
| API Development | supabase/postgres | memory-mcp, connascence-analyzer, api-docs |
| CI/CD | github, docker, aws/azure | memory-mcp, git |
| Documentation | ToC | memory-mcp, fetch |

---

## Troubleshooting

### Common Issues

**Issue**: MCP server not responding
- Check `claude_desktop_config.json` syntax
- Verify environment variables (API keys)
- Restart Claude Code

**Issue**: Connascence analyzer returns 0 violations
- Ensure tree-sitter is installed: `pip install tree-sitter tree-sitter-python`
- Check file path is correct
- Verify Python virtual environment is activated

**Issue**: Memory MCP Unicode errors on Windows
- Add `"PYTHONIOENCODING": "utf-8"` to env in config
- Avoid Unicode characters (use ASCII only)

---

## References

- **Official MCP Documentation**: https://modelcontextprotocol.io
- **MCP Servers Repository**: https://github.com/modelcontextprotocol/servers
- **Awesome MCP Servers**: https://github.com/wong2/awesome-mcp-servers
- **Connascence Analyzer**: https://github.com/DNYoussef/connascence-safety-analyzer
- **Memory MCP Triple System**: https://github.com/DNYoussef/memory-mcp-triple-system
- **ruv-SPARC Three-Loop System**: https://github.com/DNYoussef/ruv-sparc-three-loop-system

---

**Version**: 3.0.2
**Last Updated**: 2025-11-01
**Status**: Production Ready
