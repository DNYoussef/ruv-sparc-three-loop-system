# MCP Server Installation Plan

**Date**: 2025-11-01
**Goal**: Install all FREE MCP servers (no API keys, no payment required)

---

## Research Findings

### ✅ FREE Servers (Can Install)

#### 1. **MarkItDown** (Microsoft)
- **Package**: `markitdown-mcp` (Python)
- **Install**: `pip install 'markitdown[all]' && markitdown-mcp`
- **Purpose**: Convert 29+ file formats (PDF, Office, images, audio) to Markdown
- **Requirements**: Python 3.10+
- **Cost**: FREE (local processing)
- **Agents**: Documentation specialists, content processors

#### 2. **Playwright** (Microsoft)
- **Package**: `@playwright/mcp@latest`
- **Install**: `npx @playwright/mcp@latest`
- **Purpose**: Browser automation for web testing and scraping
- **Requirements**: Node.js
- **Cost**: FREE
- **Agents**: tester, web-research specialists

#### 3. **Sequential Thinking** (Anthropic)
- **Package**: `@modelcontextprotocol/server-sequential-thinking`
- **Install**: `npx -y @modelcontextprotocol/server-sequential-thinking`
- **Purpose**: Dynamic problem-solving through structured thought sequences
- **Requirements**: Node.js
- **Cost**: FREE
- **Agents**: planner, researcher, system-architect, specification, architecture

#### 4. **Fetch** (Anthropic)
- **Package**: `@modelcontextprotocol/server-fetch`
- **Install**: `npx -y @modelcontextprotocol/server-fetch`
- **Purpose**: Web content fetching and conversion for LLM usage
- **Requirements**: Node.js
- **Cost**: FREE
- **Agents**: researcher, planner

#### 5. **Filesystem** (Anthropic)
- **Package**: `@modelcontextprotocol/server-filesystem`
- **Install**: `npx -y @modelcontextprotocol/server-filesystem C:/Users/17175/claude-code-plugins`
- **Purpose**: Secure file operations with access controls
- **Requirements**: Node.js
- **Cost**: FREE
- **Agents**: ALL agents (file I/O operations)

#### 6. **Git** (Anthropic)
- **Package**: `@modelcontextprotocol/server-git`
- **Install**: `npx -y @modelcontextprotocol/server-git`
- **Purpose**: Git repository operations (read, search, manipulate)
- **Requirements**: Node.js, Git installed
- **Cost**: FREE
- **Agents**: github-modes, pr-manager, release-manager, repo-architect

#### 7. **Time** (Anthropic)
- **Package**: `@modelcontextprotocol/server-time`
- **Install**: `npx -y @modelcontextprotocol/server-time`
- **Purpose**: Time and timezone conversion
- **Requirements**: Node.js
- **Cost**: FREE
- **Agents**: Scheduling, planning, time-sensitive workflows

---

### ❌ CANNOT INSTALL (Require API Keys/Payment)

#### 1. **Context7** (Upstash)
- **Package**: `@upstash/context7-mcp`
- **Install**: `npx -y @upstash/context7-mcp --api-key YOUR_API_KEY`
- **REASON FOR EXCLUSION**: Requires Upstash API key (paid service)
- **User Constraint**: "remove all mcps that cost money ore require an api or login"

#### 2. **GitHub MCP** (Anthropic)
- **Package**: `@modelcontextprotocol/server-github`
- **Install**: Requires GITHUB_TOKEN environment variable
- **REASON FOR EXCLUSION**: Requires GitHub token (though free, still requires account/login)
- **Alternative**: Use `git` server for local operations (no API required)

#### 3. **HuggingFace**
- **Status**: No official MCP server found in research
- **REASON FOR EXCLUSION**: Package doesn't exist or requires API keys

#### 4. **DeepWiki**
- **Status**: No official MCP server found in research
- **REASON FOR EXCLUSION**: Package doesn't exist or requires authentication

#### 5. **Ref**
- **Status**: No MCP server found with this name
- **REASON FOR EXCLUSION**: Package doesn't exist

---

## Already Installed (4 Total)

1. **connascence-analyzer** - Local Python server (code quality analysis)
2. **memory-mcp** - Local Python server (persistent memory with ChromaDB)
3. **focused-changes** - Local Node.js server (file change tracking)
4. **ToC** - Local Node.js server (table of contents generation)

---

## Installation Plan

### Total Servers After Installation: 11

- **Already Installed**: 4 (local Python/Node.js servers)
- **To Install**: 7 (Anthropic + Microsoft servers)

### Installation Steps

#### Step 1: Install MarkItDown (Python)
```bash
pip install 'markitdown[all]'
pip install markitdown-mcp
```

#### Step 2: Test MarkItDown
```bash
markitdown-mcp
```

#### Step 3: Add to Claude Desktop Config
Edit `C:\Users\17175\AppData\Roaming\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "connascence-analyzer": {
      "command": "C:\\Users\\17175\\Desktop\\connascence\\venv-connascence\\Scripts\\python.exe",
      "args": ["-u", "mcp/cli.py", "mcp-server"],
      "cwd": "C:\\Users\\17175\\Desktop\\connascence",
      "env": {
        "PYTHONPATH": "C:\\Users\\17175\\Desktop\\connascence",
        "PYTHONIOENCODING": "utf-8"
      }
    },
    "memory-mcp": {
      "command": "C:\\Users\\17175\\Desktop\\memory-mcp-triple-system\\venv-memory\\Scripts\\python.exe",
      "args": ["-u", "-m", "src.mcp.stdio_server"],
      "cwd": "C:\\Users\\17175\\Desktop\\memory-mcp-triple-system",
      "env": {
        "PYTHONPATH": "C:\\Users\\17175\\Desktop\\memory-mcp-triple-system",
        "PYTHONIOENCODING": "utf-8"
      }
    },
    "focused-changes": {
      "command": "node",
      "args": ["C:/Users/17175/Documents/Cline/MCP/focused-changes-server/build/index.js"]
    },
    "toc": {
      "command": "node",
      "args": ["C:/Users/17175/Documents/Cline/MCP/toc-server/build/index.js"]
    },
    "markitdown": {
      "command": "markitdown-mcp"
    },
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@latest"]
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    },
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/Users/17175/claude-code-plugins"]
    },
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git"]
    },
    "time": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-time"]
    }
  }
}
```

#### Step 4: Restart Claude Desktop/Code

After adding all servers to the config, restart Claude Desktop/Code to load the new MCP servers.

#### Step 5: Verify Installation

Run `/mcp` command in Claude Code to list all available MCP servers.

---

## Agent-to-MCP-Server Mapping (Updated)

### Code Quality Agents (14 agents)
**Required**: memory-mcp, connascence-analyzer, focused-changes
**Recommended**: filesystem, git

**Agents**: coder, reviewer, tester, code-analyzer, functionality-audit, theater-detection-audit, production-validator, sparc-coder, analyst, backend-dev, mobile-dev, ml-developer, base-template-generator, code-review-swarm

### Research & Planning Agents
**Required**: memory-mcp
**Recommended**: fetch, sequential-thinking, time

**Agents**: researcher, planner, specification, pseudocode, architecture

### Testing Agents
**Required**: memory-mcp, focused-changes
**Recommended**: playwright

**Agents**: tester, functionality-audit

### Documentation Agents
**Required**: memory-mcp, ToC
**Recommended**: markitdown, fetch

**Agents**: api-docs, documentation specialists

### GitHub & Repository Agents
**Required**: memory-mcp
**Recommended**: git, filesystem

**Agents**: github-modes, pr-manager, release-manager, repo-architect, cicd-engineer, issue-tracker

### ALL Agents
**Required**: memory-mcp (global access)

---

## Success Criteria

- [ ] All 7 free MCP servers installed
- [ ] Claude Desktop config.json updated
- [ ] Claude Code restarted
- [ ] All servers visible in `/mcp` command
- [ ] Agent registry updated with new servers
- [ ] Documentation updated
- [ ] All references to non-existing servers removed

---

## Rollback Plan

If any server fails to install or causes issues:

1. Remove the server configuration from `claude_desktop_config.json`
2. Restart Claude Desktop/Code
3. Document the issue in this file
4. Continue with remaining servers

---

**Status**: Planning Complete
**Next Step**: Execute installation (Step 1)
