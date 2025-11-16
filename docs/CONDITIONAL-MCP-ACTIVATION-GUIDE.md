# Conditional MCP Activation Guide

**Version**: 1.0.0
**Date**: 2025-11-15
**Status**: Production Ready

---

## Overview

This guide provides step-by-step activation instructions for conditional MCP servers. These MCPs are **NOT loaded globally** to save context tokens (89.7% reduction achieved!). Instead, activate them only when specific skills require them.

## Global MCPs (Always Active - TIER 0)

These 3 MCPs are always loaded (11.3k tokens total):

```json
{
  "mcpServers": {
    "fetch": { "command": "npx", "args": ["@modelcontextprotocol/server-fetch"] },
    "sequential-thinking": { "command": "npx", "args": ["@modelcontextprotocol/server-sequential-thinking"] },
    "filesystem": { "command": "npx", "args": ["@modelcontextprotocol/server-filesystem", "C:\\Users\\17175"] }
  }
}
```

**No action needed** - these are configured in `C:\Users\17175\AppData\Roaming\Claude\claude_desktop_config.json`

---

## Conditional MCPs (Activate When Needed)

### 1. Claude Flow (12.3k tokens - 6.15% context)

**When to Activate**: SPARC methodology, swarm coordination, multi-agent workflows

**Skills That Need It**:
- sparc-methodology
- swarm-orchestration
- swarm-advanced
- when-chaining-workflows-use-cascade-orchestrator
- sop-api-development

**Activation** (PowerShell):
```powershell
# Check if already active
claude mcp list

# Add if not present
claude mcp add claude-flow npx claude-flow@alpha mcp start

# Verify
claude mcp list | Select-String "claude-flow"
```

**Deactivation**:
```powershell
# Edit config file
notepad "$env:APPDATA\Claude\claude_desktop_config.json"
# Remove "claude-flow" section, save, restart Claude Desktop
```

---

### 2. Focused Changes (1.8k tokens - 0.9% context)

**When to Activate**: Code quality tracking, change scope validation, theater detection

**Skills That Need It**:
- clarity-linter
- functionality-audit
- theater-detection-audit
- sop-dogfooding-continuous-improvement
- sop-dogfooding-pattern-retrieval
- code-review-assistant

**Activation** (PowerShell):
```powershell
# Navigate to focused-changes server directory
cd "C:\Users\17175\Documents\Cline\MCP\focused-changes-server"

# Verify server exists
if (Test-Path "build\index.js") {
    Write-Output "Server found - ready to activate"
} else {
    Write-Output "ERROR: Server not found at expected location"
}

# Add to config manually
$configPath = "$env:APPDATA\Claude\claude_desktop_config.json"
$config = Get-Content $configPath | ConvertFrom-Json
$config.mcpServers | Add-Member -NotePropertyName "focused-changes" -NotePropertyValue @{
    command = "node"
    args = @("C:\Users\17175\Documents\Cline\MCP\focused-changes-server\build\index.js")
} -Force
$config | ConvertTo-Json -Depth 10 | Set-Content $configPath

Write-Output "Focused-changes activated. Restart Claude Desktop."
```

**Deactivation**:
```powershell
$configPath = "$env:APPDATA\Claude\claude_desktop_config.json"
$config = Get-Content $configPath | ConvertFrom-Json
$config.mcpServers.PSObject.Properties.Remove("focused-changes")
$config | ConvertTo-Json -Depth 10 | Set-Content $configPath
```

---

### 3. Connascence Analyzer (5.1k tokens - 2.55% context)

**When to Activate**: Code quality analysis, coupling detection, NASA compliance checks

**Skills That Need It**:
- clarity-linter
- code-review-assistant
- sop-code-review
- sop-dogfooding-quality-detection
- production-readiness

**Activation** (PowerShell):
```powershell
# Verify Python virtual environment exists
$venvPath = "C:\Users\17175\Desktop\connascence\venv-connascence"
if (Test-Path "$venvPath\Scripts\python.exe") {
    Write-Output "Connascence analyzer venv found"
} else {
    Write-Output "ERROR: Virtual environment not found"
    exit 1
}

# Add to config manually
$configPath = "$env:APPDATA\Claude\claude_desktop_config.json"
$config = Get-Content $configPath | ConvertFrom-Json
$config.mcpServers | Add-Member -NotePropertyName "connascence-analyzer" -NotePropertyValue @{
    command = "C:\Users\17175\Desktop\connascence\venv-connascence\Scripts\python.exe"
    args = @("-u", "mcp/cli.py", "mcp-server")
    cwd = "C:\Users\17175\Desktop\connascence"
    env = @{
        PYTHONPATH = "C:\Users\17175\Desktop\connascence"
        PYTHONIOENCODING = "utf-8"
    }
} -Force
$config | ConvertTo-Json -Depth 10 | Set-Content $configPath

Write-Output "Connascence analyzer activated. Restart Claude Desktop."
```

**Deactivation**: Same pattern as focused-changes (remove from config)

---

### 4. Memory MCP (12.4k tokens - 6.2% context)

**When to Activate**: Cross-session persistence, pattern learning, multi-day projects

**Skills That Need It**:
- deep-research-orchestrator (REQUIRED for multi-month projects)
- agent-creator
- baseline-replication
- gate-validation
- sop-dogfooding-continuous-improvement
- smart-bug-fix
- sop-api-development

**Activation** (PowerShell):
```powershell
# Verify Python virtual environment
$venvPath = "C:\Users\17175\Desktop\memory-mcp-triple-system\venv-memory"
if (Test-Path "$venvPath\Scripts\python.exe") {
    Write-Output "Memory MCP venv found"
} else {
    Write-Output "ERROR: Virtual environment not found"
    exit 1
}

# Add to config
$configPath = "$env:APPDATA\Claude\claude_desktop_config.json"
$config = Get-Content $configPath | ConvertFrom-Json
$config.mcpServers | Add-Member -NotePropertyName "memory-mcp" -NotePropertyValue @{
    command = "C:\Users\17175\Desktop\memory-mcp-triple-system\venv-memory\Scripts\python.exe"
    args = @("-u", "-m", "src.mcp.stdio_server")
    cwd = "C:\Users\17175\Desktop\memory-mcp-triple-system"
    env = @{
        PYTHONPATH = "C:\Users\17175\Desktop\memory-mcp-triple-system"
        PYTHONIOENCODING = "utf-8"
        ENVIRONMENT = "development"
        LOG_LEVEL = "INFO"
    }
} -Force
$config | ConvertTo-Json -Depth 10 | Set-Content $configPath

Write-Output "Memory MCP activated. Restart Claude Desktop."
```

---

### 5. Flow-Nexus (32.5k tokens - 16.25% context)

**When to Activate**: Cloud sandboxes, neural training, distributed workflows, GitHub integration

**Skills That Need It**:
- flow-nexus-neural
- flow-nexus-platform
- flow-nexus-swarm
- feature-dev-complete
- cicd-intelligent-recovery
- deployment-readiness
- smart-bug-fix

**Activation** (PowerShell):
```powershell
# Verify Flow-Nexus is installed
npx flow-nexus@latest --version

# Add to config
claude mcp add flow-nexus npx flow-nexus@latest mcp start

# Authenticate (required for cloud features)
npx flow-nexus@latest login
# Follow prompts to create account or login

# Verify
claude mcp list | Select-String "flow-nexus"
```

**Note**: Flow-Nexus requires authentication for cloud features (sandboxes, neural training, GitHub integration). Local coordination features work without auth.

---

### 6. ruv-swarm (15.5k tokens - 7.75% context)

**When to Activate**: Advanced swarm coordination, Byzantine consensus, hierarchical topologies

**Skills That Need It**:
- swarm-advanced
- hive-mind-advanced
- when-coordinating-collective-intelligence-use-hive-mind
- sop-api-development
- sop-code-review

**Activation** (PowerShell):
```powershell
# Verify ruv-swarm is installed
npx ruv-swarm --version

# Add to config
claude mcp add ruv-swarm npx ruv-swarm mcp start

# Verify
claude mcp list | Select-String "ruv-swarm"
```

**Alternative**: Use claude-flow (already includes basic swarm features) instead of ruv-swarm unless you need advanced features like Byzantine consensus or hierarchical coordination.

---

### 7. Playwright MCP (4.2k tokens - 2.1% context)

**When to Activate**: Browser automation, accessibility testing, UI validation

**Skills That Need It**:
- wcag-accessibility
- visual-regression testing
- frontend performance analysis

**Activation** (PowerShell):
```powershell
# Install Playwright MCP
npx @modelcontextprotocol/server-playwright

# Add to config
claude mcp add playwright npx @modelcontextprotocol/server-playwright

# Verify
claude mcp list | Select-String "playwright"
```

---

### 8. TOC Generator (0.3k tokens - 0.15% context)

**When to Activate**: Documentation organization, table of contents generation

**Skills That Need It**:
- documentation skills
- knowledge-base-manager
- technical-writing-agent

**Activation** (PowerShell):
```powershell
# Verify TOC server exists
$tocPath = "C:\Users\17175\Documents\Cline\MCP\toc-server\build\index.js"
if (Test-Path $tocPath) {
    Write-Output "TOC server found"
} else {
    Write-Output "ERROR: TOC server not found"
    exit 1
}

# Add to config manually (same pattern as focused-changes)
$configPath = "$env:APPDATA\Claude\claude_desktop_config.json"
$config = Get-Content $configPath | ConvertFrom-Json
$config.mcpServers | Add-Member -NotePropertyName "toc" -NotePropertyValue @{
    command = "node"
    args = @("C:\Users\17175\Documents\Cline\MCP\toc-server\build\index.js")
} -Force
$config | ConvertTo-Json -Depth 10 | Set-Content $configPath

Write-Output "TOC generator activated. Restart Claude Desktop."
```

---

## Quick Reference: Skill → MCP Mapping

| Skill Category | Required MCPs | Token Cost | When to Activate |
|----------------|---------------|------------|------------------|
| Code Quality | connascence-analyzer, focused-changes, memory-mcp | 19.3k (9.65%) | Running code audits |
| SPARC/Swarm | claude-flow, ruv-swarm | 27.8k (13.9%) | Multi-agent workflows |
| Deep Research | memory-mcp | 12.4k (6.2%) | Multi-month projects |
| Cloud/Neural | flow-nexus | 32.5k (16.25%) | Sandbox execution, ML training |
| Frontend Testing | playwright | 4.2k (2.1%) | Browser automation |
| Documentation | toc | 0.3k (0.15%) | TOC generation |

---

## Automation Scripts

### Activate All Code Quality MCPs
```powershell
# Script: activate-code-quality-mcps.ps1
claude mcp add connascence-analyzer
claude mcp add focused-changes
claude mcp add memory-mcp
Write-Output "Code quality MCPs activated. Restart Claude Desktop."
```

### Activate All Swarm MCPs
```powershell
# Script: activate-swarm-mcps.ps1
claude mcp add claude-flow npx claude-flow@alpha mcp start
claude mcp add ruv-swarm npx ruv-swarm mcp start
Write-Output "Swarm MCPs activated. Restart Claude Desktop."
```

### Activate All Research MCPs
```powershell
# Script: activate-research-mcps.ps1
claude mcp add memory-mcp
claude mcp add fetch  # Already global, but verify
Write-Output "Research MCPs activated. Restart Claude Desktop."
```

### Reset to Minimal Config
```powershell
# Script: reset-minimal-mcps.ps1
$minimalConfig = @{
    mcpServers = @{
        fetch = @{
            command = "npx"
            args = @("@modelcontextprotocol/server-fetch")
        }
        "sequential-thinking" = @{
            command = "npx"
            args = @("@modelcontextprotocol/server-sequential-thinking")
        }
        filesystem = @{
            command = "npx"
            args = @("@modelcontextprotocol/server-filesystem", "C:\Users\17175")
        }
    }
}

$configPath = "$env:APPDATA\Claude\claude_desktop_config.json"
$minimalConfig | ConvertTo-Json -Depth 10 | Set-Content $configPath
Write-Output "Reset to minimal config (11.3k tokens). Restart Claude Desktop."
```

---

## Troubleshooting

### MCP Not Working After Activation?
1. **Restart Claude Desktop** (REQUIRED after config changes)
2. Verify config: `Get-Content "$env:APPDATA\Claude\claude_desktop_config.json" | ConvertFrom-Json | ConvertTo-Json -Depth 10`
3. Check MCP list: `claude mcp list`
4. Check for errors in Claude Desktop console

### Python MCP Servers Not Starting?
1. Verify virtual environment exists
2. Activate venv manually: `& "C:\Users\17175\Desktop\connascence\venv-connascence\Scripts\Activate.ps1"`
3. Test server: `python mcp/cli.py mcp-server`
4. Check PYTHONPATH environment variable

### Node.js MCP Servers Not Starting?
1. Verify Node.js installed: `node --version`
2. Check server file exists: `Test-Path "path\to\server\index.js"`
3. Test server manually: `node "path\to\server\index.js"`

### Token Budget Exceeded?
1. Check current usage: Skills now document token costs
2. Deactivate unused MCPs (see deactivation commands above)
3. Use minimal config and activate only what's needed
4. Monitor with: `claude mcp list` (shows all active MCPs)

---

## Best Practices

1. **Start Minimal**: Use only TIER 0 global MCPs by default
2. **Activate on Demand**: Only add MCPs when skills explicitly require them
3. **Deactivate After Use**: Remove MCPs when done to free context tokens
4. **Monitor Token Usage**: Each skill documents its MCP token costs
5. **Batch Activation**: Activate multiple related MCPs together (e.g., all code quality MCPs)
6. **Use Automation Scripts**: Create project-specific activation scripts
7. **Document Project Needs**: Track which MCPs your project requires
8. **Restart After Changes**: Always restart Claude Desktop after config changes

---

## Summary

**Default State**: 11.3k tokens (5.7% context) - Minimal global config
**Maximum State**: 109.8k tokens (54.9% context) - All MCPs loaded
**Typical Usage**: 25-40k tokens (12-20% context) - 2-3 conditional MCPs active

**Token Savings**: Up to 89.7% reduction by loading only what you need!

Each skill's MCP Requirements section has **exact activation commands** - just copy and run!