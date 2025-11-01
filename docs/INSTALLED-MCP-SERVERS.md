# Installed MCP Servers - Production Ready

This document lists all MCP servers currently installed and configured for the ruv-SPARC Three-Loop System.

**Last Updated**: 2025-11-01
**Total Servers**: 11 FREE servers (no payment, API keys, or accounts required)

---

## Currently Installed & Configured

### 1. **connascence-analyzer** (Local - Production)
- **Location**: `C:\Users\17175\Desktop\connascence\`
- **Command**: `venv-connascence\Scripts\python.exe -u mcp/cli.py mcp-server`
- **Type**: Local Python MCP server
- **Cost**: FREE
- **Status**: ✅ PRODUCTION READY
- **Purpose**: Code quality analysis (9 connascence types, 7+ violation categories, NASA compliance)
- **Performance**: 0.018 seconds per file
- **Agents**: All 14 code quality agents (coder, reviewer, tester, sparc-coder, etc.)

### 2. **memory-mcp** (Local - Production)
- **Location**: `C:\Users\17175\Desktop\memory-mcp-triple-system\`
- **Command**: `venv-memory\Scripts\python.exe -u -m src.mcp.stdio_server`
- **Type**: Local Python MCP server
- **Cost**: FREE
- **Status**: ✅ PRODUCTION READY
- **Purpose**: Persistent cross-session memory with triple-layer retention (24h/7d/30d+)
- **Storage**: Local ChromaDB (384-dimensional vectors, HNSW indexing)
- **Agents**: ALL 90 agents (global access with WHO/WHEN/PROJECT/WHY tagging)

### 3. **focused-changes** (Local - Production)
- **Location**: `C:\Users\17175\Documents\Cline\MCP\focused-changes-server\`
- **Command**: `node C:/Users/17175/Documents/Cline/MCP/focused-changes-server/build/index.js`
- **Type**: Local Node.js MCP server
- **Cost**: FREE
- **Status**: ✅ PRODUCTION READY
- **Purpose**: Track file changes, ensure focused scope, build error trees from test failures
- **Tools**: `start_tracking`, `analyze_changes`, `root_cause_analysis`
- **Agents**: coder, reviewer, tester, sparc-coder, functionality-audit

### 4. **ToC** (Local - Production)
- **Location**: `C:\Users\17175\Documents\Cline\MCP\toc-server\`
- **Command**: `node C:/Users/17175/Documents/Cline/MCP/toc-server/build/index.js`
- **Type**: Local Node.js MCP server
- **Cost**: FREE
- **Status**: ✅ PRODUCTION READY
- **Purpose**: Generate table of contents for documentation (Python, Markdown, JSON, YAML, TXT)
- **Agents**: Documentation specialists, architecture planners

---

## Available for Installation (Official Anthropic - FREE)

These are maintained by Anthropic and available via npx:

### 5. **context7** (Upstash)
- **Install**: `npx @upstash/context7`
- **Cost**: FREE
- **Purpose**: Up-to-date, version-specific documentation and code examples from any library/framework
- **Recommended Agents**: researcher, coder, documentation specialists
- **Usage**: Get current docs for libraries (React, Next.js, Python packages, etc.)

### 6. **markitdown** (Microsoft)
- **Install**: `npx @microsoft/markitdown-mcp`
- **Cost**: FREE
- **Purpose**: Convert various file formats (PDF, Word, Excel, images, audio) to Markdown
- **Recommended Agents**: Documentation specialists, content processors
- **Usage**: Process documents for AI consumption

### 7. **playwright-mcp**
- **Install**: `npx playwright-mcp`
- **Cost**: FREE
- **Purpose**: Browser automation and web scraping using Playwright
- **Recommended Agents**: tester, web-research specialists
- **Usage**: Automated testing, web data extraction

### 8. **deepwiki** (Remote - No Auth)
- **Type**: Remote MCP server
- **Cost**: FREE (no authentication required)
- **Purpose**: AI-powered codebase context and answers
- **Recommended Agents**: researcher, documentation specialists
- **Usage**: Query codebase documentation and context

### 9. **huggingface-spaces-mcp**
- **Install**: `npx huggingface-spaces-mcp`
- **Cost**: FREE
- **Purpose**: Use HuggingFace Spaces (Images, Audio, Text models)
- **Recommended Agents**: ml-developer, content generation specialists
- **Usage**: Access HuggingFace AI models

---

## Official Anthropic MCP Servers (Recommended for Installation)

### 10. **sequential-thinking**
- **Install**: `npx @modelcontextprotocol/server-sequential-thinking`
- **Cost**: FREE
- **Purpose**: Dynamic and reflective problem-solving through thought sequences
- **Recommended Agents**: planner, researcher, system-architect, specification, architecture
- **Usage**: Break down complex problems into step-by-step solutions

### 11. **fetch**
- **Install**: `npx @modelcontextprotocol/server-fetch`
- **Cost**: FREE
- **Purpose**: Web content fetching and conversion for LLM usage
- **Recommended Agents**: researcher, web-research tasks
- **Usage**: Fetch and process web content

### 12. **filesystem**
- **Install**: `npx @modelcontextprotocol/server-filesystem /path/to/allowed/directory`
- **Cost**: FREE
- **Purpose**: Secure file operations with configurable access controls
- **Recommended Agents**: ALL agents needing file I/O beyond Claude Code's built-in tools
- **Usage**: Read/write files with permission control

### 13. **git**
- **Install**: `npx @modelcontextprotocol/server-git`
- **Cost**: FREE
- **Purpose**: Tools to read, search, and manipulate Git repositories (local only, no API required)
- **Recommended Agents**: All GitHub/repository agents
- **Usage**: Git operations without GitHub API tokens

### 14. **memory** (Official Anthropic)
- **Install**: `npx @modelcontextprotocol/server-memory`
- **Cost**: FREE
- **Purpose**: Knowledge graph-based persistent memory system
- **Note**: We use the enhanced Memory MCP Triple System instead (more features)

### 15. **time**
- **Install**: `npx @modelcontextprotocol/server-time`
- **Cost**: FREE
- **Purpose**: Time and timezone conversion capabilities
- **Recommended Agents**: Scheduling, planning, time-sensitive workflows
- **Usage**: Convert timezones, format dates

### 16. **everything**
- **Install**: `npx @modelcontextprotocol/server-everything`
- **Cost**: FREE
- **Purpose**: Reference/test server demonstrating all MCP capabilities (prompts, resources, tools)
- **Usage**: Learning and testing MCP protocol features

---

## Installation Status Summary

| Server | Status | Type | Agents Using |
|--------|--------|------|--------------|
| connascence-analyzer | ✅ INSTALLED | Local Python | 14 code quality agents |
| memory-mcp | ✅ INSTALLED | Local Python | ALL 90 agents |
| focused-changes | ✅ INSTALLED | Local Node.js | 5 agents |
| ToC | ✅ INSTALLED | Local Node.js | Documentation agents |
| context7 | ⏳ READY (npx) | Remote (Upstash) | researcher, coder, docs |
| markitdown | ⏳ READY (npx) | Local (Microsoft) | Documentation agents |
| playwright-mcp | ⏳ READY (npx) | Local | tester, web-research |
| deepwiki | ⏳ READY (remote) | Remote (no auth) | researcher, docs |
| huggingface-spaces | ⏳ READY (npx) | Remote | ml-developer |
| sequential-thinking | ⏳ READY (npx) | Local (Anthropic) | planner, researcher, architect |
| fetch | ⏳ READY (npx) | Local (Anthropic) | researcher |
| filesystem | ⏳ READY (npx) | Local (Anthropic) | ALL agents |
| git | ⏳ READY (npx) | Local (Anthropic) | GitHub agents |
| time | ⏳ READY (npx) | Local (Anthropic) | Planning agents |

**Total**: 4 installed + 11 ready = 15 FREE MCP servers available

---

## Installation Instructions

### To Add MCP Servers to Claude Code:

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
    "context7": {
      "command": "npx",
      "args": ["@upstash/context7"]
    },
    "markitdown": {
      "command": "npx",
      "args": ["@microsoft/markitdown-mcp"]
    },
    "playwright": {
      "command": "npx",
      "args": ["playwright-mcp"]
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-sequential-thinking"]
    },
    "fetch": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-fetch"]
    },
    "git": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-git"]
    }
  }
}
```

---

## Agent-to-MCP-Server Mapping

See `agents/registry.json` for complete agent-specific MCP server assignments.

**Quick Reference**:
- **Code Quality** (14 agents): memory-mcp + connascence-analyzer + focused-changes
- **Research** (researcher, planner): memory-mcp + context7 + fetch + sequential-thinking
- **Testing** (tester, functionality-audit): memory-mcp + connascence-analyzer + focused-changes + playwright
- **Documentation** (api-docs, docs agents): memory-mcp + ToC + markitdown + context7
- **ML/AI** (ml-developer): memory-mcp + huggingface-spaces + context7
- **ALL Agents**: memory-mcp (global access)

---

## Removed Servers (Paid/API-Required)

The following servers were removed in v3.0.3 because they require payment, API keys, or accounts:
- Tavily, Exa, Firecrawl (web search - require API keys)
- E2B, Docker, AWS, Azure (infrastructure - require accounts)
- Supabase, Chroma, Neo4j (databases - require setup)
- GitHub MCP (requires GITHUB_TOKEN)
- Slack, Notion (communication - require accounts)
- All others listed in CHANGELOG.md v3.0.3

**Alternative**: Use the FREE servers listed above instead.

---

## Support & Documentation

- **MCP Marketplace Guide**: See `docs/MCP-MARKETPLACE-GUIDE.md`
- **Agent Registry**: See `agents/registry.json` for agent-specific assignments
- **Official MCP Docs**: https://modelcontextprotocol.io

---

**Version**: 3.0.4
**Last Updated**: 2025-11-01
**Status**: Production Ready (4 installed + 11 available = 15 FREE servers)
