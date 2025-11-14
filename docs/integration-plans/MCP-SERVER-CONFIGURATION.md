# MCP Server Configuration Guide

**Project**: Obsidian Memory + Connascence Analyzer Integration
**Version**: 1.0.0
**Created**: 2025-11-01
**Status**: Phase 1 & 2 Configuration Documentation

---

## Overview

This guide provides complete configuration instructions for deploying two MCP (Model Context Protocol) servers for Claude Code integration:

1. **Memory MCP Triple System** - Persistent memory with Obsidian vault synchronization
2. **Connascence Safety Analyzer** - Code coupling analysis

Both servers will be registered in Claude Desktop's MCP configuration and accessible to Claude Code sessions.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Memory MCP Server Setup](#memory-mcp-server-setup)
3. [Connascence MCP Server Setup](#connascence-mcp-server-setup)
4. [Claude Desktop MCP Configuration](#claude-desktop-mcp-configuration)
5. [Testing & Verification](#testing--verification)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tuning](#performance-tuning)

---

## Prerequisites

### System Requirements

- **OS**: Windows 10/11 (current user environment)
- **Python**: 3.12+ (required for both MCP servers)
- **Node.js**: v20.17.0 (already installed)
- **Claude Desktop**: Latest version with MCP support

### Directory Structure

```
C:\Users\17175\
├── Desktop\
│   ├── memory-mcp-triple-system\     ← Memory MCP source
│   └── connascence\                   ← Connascence analyzer source
├── Obsidian\
│   └── 12FA-Memory\                   ← Obsidian vault (will create)
└── .config\
    └── claude\
        └── mcp_servers.json           ← MCP server registry (will create/update)
```

### Python Environment Setup

```bash
# Install Python 3.12+ if not already installed
# Verify version
python --version  # Should be 3.12+

# Create virtual environments for isolation
cd C:\Users\17175\Desktop\memory-mcp-triple-system
python -m venv venv-memory

cd C:\Users\17175\Desktop\connascence
python -m venv venv-connascence
```

---

## Memory MCP Server Setup

### Step 1: Install Dependencies

```bash
cd C:\Users\17175\Desktop\memory-mcp-triple-system

# Activate virtual environment
.\venv-memory\Scripts\activate

# Install core dependencies
pip install fastapi uvicorn chromadb hnswlib python-dotenv pydantic

# Install Obsidian integration
pip install markdown frontmatter watchdog

# Install testing dependencies (optional but recommended)
pip install pytest pytest-asyncio pytest-cov

# Verify installation
pip list
```

**CRITICAL WINDOWS ISSUE**: `hnswlib-node` native bindings

The Memory MCP system uses `hnswlib` for vector search. On Windows, native bindings may fail to compile.

**Solutions** (in order of preference):

1. **Pre-compiled binaries** (recommended):
```bash
pip install hnswlib --only-binary :all:
```

2. **WSL2 Docker fallback** (if #1 fails):
```bash
# Install Docker Desktop with WSL2 backend
# Run Memory MCP in Docker container
docker run -v C:\Users\17175\Desktop\memory-mcp-triple-system:/app \
           -v C:\Users\17175\Obsidian\12FA-Memory:/vault \
           -p 8000:8000 \
           python:3.12-slim \
           bash -c "pip install -r /app/requirements.txt && python -m src.mcp.server"
```

3. **Build tools** (last resort):
```bash
# Install Microsoft Visual C++ 14.0+ Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/
# Then retry: pip install hnswlib
```

### Step 2: Configure Environment

Create `.env` file in `C:\Users\17175\Desktop\memory-mcp-triple-system\`:

```env
# Memory MCP Configuration

# Server Settings
MCP_SERVER_HOST=127.0.0.1
MCP_SERVER_PORT=8000
MCP_TRANSPORT=stdio

# Memory Retention Policies (in seconds)
SHORT_TERM_RETENTION=86400      # 24 hours
MID_TERM_RETENTION=604800       # 7 days
LONG_TERM_RETENTION=2592000     # 30 days

# Obsidian Vault Integration
OBSIDIAN_VAULT_PATH=C:\Users\17175\Obsidian\12FA-Memory
OBSIDIAN_SYNC_ENABLED=true
OBSIDIAN_SYNC_INTERVAL=300      # 5 minutes
OBSIDIAN_READ_ONLY=true         # Start with read-only for safety

# ChromaDB Settings
CHROMA_DB_PATH=./data/chroma
CHROMA_COLLECTION_NAME=memory_mcp
CHROMA_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Mode Detection
MODE_DETECTION_ENABLED=true
MODE_PATTERNS_FILE=./config/mode-patterns.yaml

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/memory-mcp.log

# Performance
MAX_MEMORY_ENTRIES=10000
QUERY_CACHE_SIZE=1000
QUERY_CACHE_TTL=3600            # 1 hour
```

### Step 3: Create Obsidian Vault

```bash
# Create vault directory
mkdir -p C:\Users\17175\Obsidian\12FA-Memory

# Create vault structure
cd C:\Users\17175\Obsidian\12FA-Memory
mkdir -p {short-term,mid-term,long-term,patterns,dashboard}

# Create .obsidian config
mkdir .obsidian
```

Create `C:\Users\17175\Obsidian\12FA-Memory\.obsidian\config`:

```json
{
  "vimMode": false,
  "theme": "obsidian",
  "pluginEnabledStatus": {
    "graph": true,
    "dataview": true
  },
  "defaultViewMode": "source",
  "foldHeading": true,
  "foldIndent": true
}
```

### Step 4: Start Memory MCP Server

```bash
cd C:\Users\17175\Desktop\memory-mcp-triple-system
.\venv-memory\Scripts\activate

# Start server
python -m src.mcp.server

# Server should output:
# INFO:     Started MCP server on stdio
# INFO:     Memory MCP Triple System v1.0.0 ready
# INFO:     Obsidian vault: C:\Users\17175\Obsidian\12FA-Memory
# INFO:     Mode detection: 29 patterns loaded
```

**Health Check**:

```bash
# In a new terminal (keep server running)
curl http://127.0.0.1:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "obsidian_vault": "C:\\Users\\17175\\Obsidian\\12FA-Memory",
#   "total_entries": 0,
#   "short_term": 0,
#   "mid_term": 0,
#   "long_term": 0
# }
```

---

## Connascence MCP Server Setup

### Step 1: Install Dependencies

```bash
cd C:\Users\17175\Desktop\connascence

# Activate virtual environment
.\venv-connascence\Scripts\activate

# Install core dependencies
pip install fastapi uvicorn pydantic ast-parser

# Install analysis dependencies
pip install networkx graphviz python-magic

# Install testing dependencies
pip install pytest pytest-asyncio

# Verify installation
pip list
```

### Step 2: Configure Environment

Create `.env` file in `C:\Users\17175\Desktop\connascence\`:

```env
# Connascence MCP Configuration

# Server Settings
MCP_SERVER_HOST=127.0.0.1
MCP_SERVER_PORT=8001            # Different port from Memory MCP
MCP_TRANSPORT=stdio

# Analysis Settings
SUPPORTED_LANGUAGES=javascript,typescript
ANALYSIS_PROFILE=nasa-compliance
INCREMENTAL_CACHE_ENABLED=true
CACHE_PATH=./data/cache

# Workspace Configuration
DEFAULT_WORKSPACE=C:\Users\17175
EXCLUDE_PATTERNS=node_modules,dist,build,.git,venv
INCLUDE_EXTENSIONS=.js,.ts,.jsx,.tsx

# Connascence Types (all 9 enabled)
ENABLE_CON=true    # Name
ENABLE_COT=true    # Type
ENABLE_COM=true    # Meaning
ENABLE_COP=true    # Position
ENABLE_COA=true    # Algorithm
ENABLE_COE=true    # Execution
ENABLE_COV=true    # Value
ENABLE_COI=true    # Identity
ENABLE_COID=true   # Identity of reference

# Performance
MAX_FILE_SIZE_MB=10
PARALLEL_WORKERS=4
ANALYSIS_TIMEOUT_SECONDS=60

# Output
SARIF_OUTPUT_ENABLED=true
SARIF_OUTPUT_PATH=./reports/sarif

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/connascence-mcp.log
```

### Step 3: Create Workspace Configuration

Create `.connascenceignore` in `C:\Users\17175\`:

```gitignore
# Connascence Analysis - Ignore Patterns

# Dependencies
node_modules/
venv/
venv-*/
__pycache__/

# Build artifacts
dist/
build/
out/
.next/

# Version control
.git/
.svn/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Test coverage
coverage/
.nyc_output/

# Large data
*.db
*.sqlite
*.db-journal
```

### Step 4: Start Connascence MCP Server

```bash
cd C:\Users\17175\Desktop\connascence
.\venv-connascence\Scripts\activate

# Start server
python -m src.mcp.server

# Server should output:
# INFO:     Started MCP server on stdio
# INFO:     Connascence Safety Analyzer v1.0.0 ready
# INFO:     Analysis profile: nasa-compliance
# INFO:     Supported languages: javascript, typescript
# INFO:     All 9 connascence types enabled
```

**Health Check**:

```bash
# In a new terminal (keep server running)
curl http://127.0.0.1:8001/health

# Expected response:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "profile": "nasa-compliance",
#   "languages": ["javascript", "typescript"],
#   "connascence_types": 9,
#   "cache_enabled": true
# }
```

---

## Claude Desktop MCP Configuration

### Step 1: Locate MCP Configuration File

**Path**: `C:\Users\17175\.config\claude\mcp_servers.json`

If the file doesn't exist, create it:

```bash
mkdir -p C:\Users\17175\.config\claude
touch C:\Users\17175\.config\claude\mcp_servers.json
```

### Step 2: Add MCP Server Entries

Edit `C:\Users\17175\.config\claude\mcp_servers.json`:

```json
{
  "mcpServers": {
    "memory-mcp": {
      "command": "python",
      "args": [
        "-m",
        "src.mcp.server"
      ],
      "cwd": "C:\\Users\\17175\\Desktop\\memory-mcp-triple-system",
      "env": {
        "PYTHONPATH": "C:\\Users\\17175\\Desktop\\memory-mcp-triple-system",
        "OBSIDIAN_VAULT_PATH": "C:\\Users\\17175\\Obsidian\\12FA-Memory"
      },
      "transport": "stdio",
      "description": "3-layer persistent memory with Obsidian vault synchronization",
      "enabled": true
    },
    "connascence-analyzer": {
      "command": "python",
      "args": [
        "-m",
        "src.mcp.server"
      ],
      "cwd": "C:\\Users\\17175\\Desktop\\connascence",
      "env": {
        "PYTHONPATH": "C:\\Users\\17175\\Desktop\\connascence",
        "DEFAULT_WORKSPACE": "C:\\Users\\17175"
      },
      "transport": "stdio",
      "description": "Comprehensive code coupling analysis (9 connascence types)",
      "enabled": true
    },
    "claude-flow": {
      "command": "npx",
      "args": [
        "claude-flow@alpha",
        "mcp",
        "start"
      ],
      "transport": "stdio",
      "description": "claude-flow coordination and hot memory",
      "enabled": true
    },
    "ruv-swarm": {
      "command": "npx",
      "args": [
        "ruv-swarm",
        "mcp",
        "start"
      ],
      "transport": "stdio",
      "description": "ruv-swarm enhanced coordination (optional)",
      "enabled": false
    },
    "flow-nexus": {
      "command": "npx",
      "args": [
        "flow-nexus@latest",
        "mcp",
        "start"
      ],
      "transport": "stdio",
      "description": "flow-nexus cloud features (optional)",
      "enabled": false
    }
  }
}
```

### Step 3: Validate Configuration

```bash
# Validate JSON syntax
cat C:\Users\17175\.config\claude\mcp_servers.json | python -m json.tool

# Should output formatted JSON with no errors
```

### Step 4: Restart Claude Desktop

1. Quit Claude Desktop completely
2. Relaunch Claude Desktop
3. Check status bar for MCP server indicators

**Expected Indicators**:
- 🟢 `memory-mcp` (green = connected)
- 🟢 `connascence-analyzer` (green = connected)
- 🟢 `claude-flow` (green = connected)

---

## Testing & Verification

### Test 1: Memory MCP Basic Operations

In Claude Code, run:

```javascript
// Test 1: Store short-term memory
await memory.store({
  key: "test-1",
  value: "Hello from Memory MCP",
  namespace: "testing",
  retention_policy: "short-term"
});

// Test 2: Retrieve memory
const result = await memory.retrieve({
  key: "test-1",
  namespace: "testing"
});
console.log(result);  // Should output: "Hello from Memory MCP"

// Test 3: Query by semantic similarity
const results = await memory.query({
  query: "memory test",
  namespace: "testing",
  limit: 5
});
console.log(results);
```

**Expected Results**:
- ✅ Store operation succeeds
- ✅ Retrieve returns correct value
- ✅ Query returns semantically similar entries

### Test 2: Memory MCP Mode Detection

```javascript
// Test mode detection
const mode = await memory.detect_mode({
  context: "I need to plan a new feature for user authentication"
});
console.log(mode);  // Should output: "planning"

const mode2 = await memory.detect_mode({
  context: "Run the build command and fix any errors"
});
console.log(mode2);  // Should output: "execution"
```

**Expected Results**:
- ✅ Planning mode detected for design tasks
- ✅ Execution mode detected for coding tasks
- ✅ Brainstorming mode detected for exploration tasks

### Test 3: Obsidian Vault Sync

```bash
# Check that memory entries appear in Obsidian vault
ls C:\Users\17175\Obsidian\12FA-Memory\short-term\

# Should see markdown files created by Memory MCP
# Example: test-1.md

cat C:\Users\17175\Obsidian\12FA-Memory\short-term\test-1.md
# Should contain:
# ---
# key: test-1
# namespace: testing
# retention: short-term
# created: 2025-11-01T...
# ---
# Hello from Memory MCP
```

**Expected Results**:
- ✅ Files created in short-term directory
- ✅ Frontmatter contains metadata
- ✅ Content matches stored value

### Test 4: Connascence Analysis

In Claude Code, run:

```javascript
// Test 1: Analyze a single file
const result = await connascence.analyze_file({
  file_path: "C:\\Users\\17175\\example.js"
});
console.log(result);

// Test 2: Analyze workspace
const workspace_results = await connascence.analyze_workspace({
  workspace_path: "C:\\Users\\17175\\test-project",
  incremental: true
});
console.log(workspace_results);

// Test 3: Health check
const health = await connascence.health_check();
console.log(health);
```

**Expected Results**:
- ✅ Single file analysis completes in <5s
- ✅ Workspace analysis uses incremental caching
- ✅ All 9 connascence types detected
- ✅ 0% false positives (verify against known violations)

### Test 5: Theater→Connascence Pipeline

```javascript
// This will be implemented in Phase 2 (P2-T4)
// Post-edit hook should automatically trigger:
// 1. Theater detection (existing)
// 2. Connascence analysis (if theater check passes)

// Test by editing a file with intentional coupling issue
// Expected: Both theater and connascence results stored in memory
```

---

## Troubleshooting

### Issue 1: Memory MCP Server Won't Start

**Symptoms**:
- Server crashes immediately
- "Could not locate bindings file" error (hnswlib-node)

**Solutions**:

1. **Check Python version**:
```bash
python --version  # Must be 3.12+
```

2. **Reinstall dependencies with pre-compiled binaries**:
```bash
cd C:\Users\17175\Desktop\memory-mcp-triple-system
.\venv-memory\Scripts\activate
pip uninstall hnswlib -y
pip install hnswlib --only-binary :all:
```

3. **Use WSL2 Docker fallback** (see Step 1 above)

4. **Check Obsidian vault path**:
```bash
# Ensure vault directory exists
mkdir -p C:\Users\17175\Obsidian\12FA-Memory
```

### Issue 2: Connascence MCP Server Port Conflict

**Symptoms**:
- "Port 8001 already in use"

**Solutions**:

1. **Change port in .env**:
```env
MCP_SERVER_PORT=8002  # Or any available port
```

2. **Kill existing process**:
```bash
# Find process using port 8001
netstat -ano | findstr :8001

# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Issue 3: Claude Desktop Not Detecting MCP Servers

**Symptoms**:
- No MCP server indicators in Claude Desktop
- Claude Code can't access MCP tools

**Solutions**:

1. **Validate JSON syntax**:
```bash
cat C:\Users\17175\.config\claude\mcp_servers.json | python -m json.tool
```

2. **Check server status**:
```bash
# Test memory-mcp manually
cd C:\Users\17175\Desktop\memory-mcp-triple-system
.\venv-memory\Scripts\activate
python -m src.mcp.server

# Should start without errors
```

3. **Restart Claude Desktop completely**:
- Quit Claude Desktop (not just close window)
- Wait 5 seconds
- Relaunch

4. **Check Claude Desktop logs**:
```bash
# Windows logs location
%APPDATA%\Claude\logs\

# Look for MCP-related errors
```

### Issue 4: Obsidian Sync Not Working

**Symptoms**:
- Memory entries not appearing in Obsidian vault
- Sync errors in logs

**Solutions**:

1. **Check vault path in .env**:
```env
OBSIDIAN_VAULT_PATH=C:\Users\17175\Obsidian\12FA-Memory  # Must match actual path
```

2. **Check permissions**:
```bash
# Ensure vault directory is writable
icacls C:\Users\17175\Obsidian\12FA-Memory
```

3. **Enable sync in config**:
```env
OBSIDIAN_SYNC_ENABLED=true
OBSIDIAN_READ_ONLY=false  # If you want bidirectional sync
```

4. **Check logs**:
```bash
tail -f C:\Users\17175\Desktop\memory-mcp-triple-system\logs\memory-mcp.log
```

### Issue 5: Connascence Analysis Timeout

**Symptoms**:
- Analysis takes >60s
- Timeout errors

**Solutions**:

1. **Enable incremental caching**:
```env
INCREMENTAL_CACHE_ENABLED=true
```

2. **Increase timeout**:
```env
ANALYSIS_TIMEOUT_SECONDS=120
```

3. **Exclude large directories**:
```env
EXCLUDE_PATTERNS=node_modules,dist,build,.git,venv,data,logs
```

4. **Increase parallel workers**:
```env
PARALLEL_WORKERS=8  # Use more CPU cores
```

---

## Performance Tuning

### Memory MCP Optimization

**Target**: <200ms query time (P1 NFR)

```env
# .env optimizations

# Enable LRU cache
QUERY_CACHE_SIZE=1000
QUERY_CACHE_TTL=3600

# Batch Obsidian sync
OBSIDIAN_SYNC_INTERVAL=300  # 5 minutes instead of real-time

# Limit entries
MAX_MEMORY_ENTRIES=10000
```

**Code-level optimization** (implemented in Phase 4):

```python
# src/mcp/memory_cache.py
from functools import lru_cache

@lru_cache(maxsize=1000)
def retrieve_memory(key: str, namespace: str):
    # Cached retrieval for frequent queries
    pass
```

### Connascence Optimization

**Target**: <5s cached, <15s workspace (P2 NFR)

```env
# .env optimizations

# Enable incremental caching
INCREMENTAL_CACHE_ENABLED=true

# Parallel workers
PARALLEL_WORKERS=8

# Exclude patterns
EXCLUDE_PATTERNS=node_modules,dist,build,.git,venv,__pycache__,coverage
```

**Incremental caching strategy** (implemented in Phase 4):

```python
# src/mcp/incremental_cache.py
def analyze_workspace_incremental(workspace_path: str):
    # Only re-analyze files changed since last run
    # Track file dependencies, invalidate cascade
    pass
```

### Monitoring Performance

```bash
# Memory MCP metrics
curl http://127.0.0.1:8000/metrics

# Expected output:
# {
#   "avg_query_time_ms": 150,
#   "cache_hit_rate": 0.82,
#   "total_entries": 5432,
#   "obsidian_sync_lag_seconds": 240
# }

# Connascence metrics
curl http://127.0.0.1:8001/metrics

# Expected output:
# {
#   "avg_analysis_time_cached_ms": 3200,
#   "avg_analysis_time_workspace_ms": 12400,
#   "cache_hit_rate": 0.91,
#   "violations_detected": 5743,
#   "false_positive_rate": 0.0
# }
```

---

## Next Steps

After completing MCP server configuration:

1. **Phase 1**: Test 3-layer memory retention (P1-T4)
2. **Phase 1**: Validate mode detection (P1-T7)
3. **Phase 2**: Create theater→connascence pipeline hook (P2-T4)
4. **Phase 2**: Test all 9 connascence types (P2-T6)
5. **Phase 3**: Implement learning loop (P3-T1 through P3-T6)

---

## Configuration Files Summary

| File | Location | Purpose |
|------|----------|---------|
| `mcp_servers.json` | `C:\Users\17175\.config\claude\` | Claude Desktop MCP registry |
| `.env` (Memory) | `C:\Users\17175\Desktop\memory-mcp-triple-system\` | Memory MCP configuration |
| `.env` (Connascence) | `C:\Users\17175\Desktop\connascence\` | Connascence configuration |
| `.connascenceignore` | `C:\Users\17175\` | Connascence exclude patterns |
| `config` (Obsidian) | `C:\Users\17175\Obsidian\12FA-Memory\.obsidian\` | Obsidian vault settings |

---

**Version**: 1.0.0
**Created**: 2025-11-01
**Status**: Phase 1 & 2 Configuration Complete
**Next**: Testing & Verification → Phase 3 Learning Loop
