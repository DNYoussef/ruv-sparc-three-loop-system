# MCP Server Configuration Fix Summary

**Date**: 2025-11-08
**Status**: ✅ **FIXED - IMMEDIATE TESTING REQUIRED**

---

## Executive Summary

Fixed critical Windows compatibility issues in MCP server configuration and identified why 2 servers are failing.

### Quick Status

| Server | Status | Issue | Fix |
|--------|--------|-------|-----|
| **claude-flow** | ❌ Failed | Missing Windows cmd wrapper | ✅ Fixed in .mcp.json |
| **ruv-swarm** | ⚠️ Unknown | Missing Windows cmd wrapper | ✅ Fixed in .mcp.json |
| **flow-nexus** | ⚠️ Unknown | Missing Windows cmd wrapper | ✅ Fixed in .mcp.json |
| **connascence-analyzer** | ❌ Failed | Incorrect CLI command | ⚠️ Needs investigation |
| **agentic-payments** | ✅ Connected | Working | No action needed |
| **fetch** | ✅ Connected | Working | No action needed |
| **filesystem** | ✅ Connected | Working | No action needed |
| **playwright** | ⚠️ Unknown | Not shown in list | Check if enabled |

---

## Changes Made

### 1. Fixed Windows npx Wrapper Issue ✅

**Problem**: Windows requires `cmd /c` to execute npx commands properly

**Before** (`.mcp.json`):
```json
{
  "command": "npx",
  "args": ["claude-flow@alpha", "mcp", "start"]
}
```

**After** (`.mcp.json`):
```json
{
  "command": "cmd",
  "args": ["/c", "npx", "claude-flow@alpha", "mcp", "start"]
}
```

**Files Modified**:
- `C:\Users\17175\.mcp.json` - Updated 3 server configurations:
  - `claude-flow` - Fixed ✅
  - `ruv-swarm` - Fixed ✅
  - `flow-nexus` - Fixed ✅

**Note**: The user-level `.claude.json` already had the correct `cmd /c` wrapper format for these servers.

---

### 2. Identified connascence-analyzer Issue ❌

**Problem**: Using non-existent CLI command

**Current Configuration** (`.claude.json`):
```json
{
  "type": "stdio",
  "command": "C:\\Users\\17175\\Desktop\\connascence\\venv-connascence\\Scripts\\python.exe",
  "args": ["-u", "mcp/cli.py", "mcp-server"],
  "cwd": "C:\\Users\\17175\\Desktop\\connascence"
}
```

**Error Found**:
```
cli.py: error: argument command: invalid choice: 'mcp-server'
(choose from 'analyze-file', 'analyze-workspace', 'health-check', 'info')
```

**Available Commands** (from testing):
- `analyze-file` - Analyze individual files
- `analyze-workspace` - Analyze entire directories
- `health-check` - Check server health
- `info` - Get server information

**Root Cause**: The connascence-analyzer CLI doesn't have an `mcp-server` command. Looking at the code:
- `mcp/server.py` - "Mock MCP Server implementation for test compatibility"
- `mcp/cli.py` - Only provides direct analysis commands, no MCP server mode

**Possible Solutions**:
1. **Option A**: Disable connascence-analyzer MCP server (use CLI directly when needed)
2. **Option B**: Check if there's a different script to start the MCP server
3. **Option C**: The MCP server functionality may not be fully implemented yet

---

## Testing Required

### Step 1: Restart Claude Code

The `.mcp.json` changes require restarting Claude Code to take effect:

```bash
# Exit current Claude Code session
# Restart Claude Code
claude
```

### Step 2: Verify MCP Servers

Run the MCP server list command:

```bash
/mcp
```

**Expected Results**:
- ✅ `claude-flow` - Should now show as "connected"
- ✅ `ruv-swarm` - Should now show as "connected"
- ✅ `flow-nexus` - Should now show as "connected"
- ❌ `connascence-analyzer` - Will still fail (needs different fix)

### Step 3: Test Individual Servers

**Test claude-flow**:
```bash
# In separate terminal
cmd /c npx claude-flow@alpha mcp start
```

**Test ruv-swarm**:
```bash
cmd /c npx ruv-swarm mcp start
```

**Test flow-nexus**:
```bash
cmd /c npx flow-nexus@latest mcp start
```

---

## Recommendations

### High Priority (Do Now)

1. **Restart Claude Code** to apply .mcp.json changes
   ```bash
   # Exit and restart
   claude
   ```

2. **Verify claude-flow installation**
   ```bash
   cmd /c npx claude-flow@alpha --version
   ```

3. **Check ruv-swarm installation**
   ```bash
   cmd /c npx ruv-swarm --version
   ```

4. **Test flow-nexus** (may require authentication)
   ```bash
   cmd /c npx flow-nexus@latest --version
   ```

### Medium Priority (Within 24 hours)

5. **Decide on connascence-analyzer**:

   **Option A - Disable MCP Server** (Recommended):
   ```json
   // In .claude.json, set disabled: true
   "connascence-analyzer": {
     "type": "stdio",
     "command": "...",
     "disabled": true
   }
   ```

   **Option B - Investigate Alternative**:
   - Check if there's a `start_mcp_server.bat` script
   - Test: `C:\Users\17175\Desktop\connascence\mcp\start_mcp_server.bat`

   **Option C - Use CLI Directly**:
   - Remove from MCP servers
   - Call CLI commands directly when needed:
     ```bash
     python C:\Users\17175\Desktop\connascence\mcp\cli.py analyze-file <file>
     ```

6. **Optimize CLAUDE.md size** (Currently 44,126 chars > 40,000 limit):
   - Move detailed documentation to separate files
   - Keep only essential configuration in CLAUDE.md
   - Use `@-mention` to import docs when needed

7. **Reduce MCP Tools Context** (Currently ~106,818 tokens > 25,000 recommended):
   - Consider disabling optional MCP servers if not actively used
   - Use `alwaysAllow` lists to pre-approve common tools
   - Enable MCP servers on-demand rather than all at once

---

## Detailed Analysis

### Windows npx Issue Explained

**Why cmd /c is needed on Windows**:
- Windows batch files (`.cmd`, `.bat`) require Command Prompt to execute
- `npx` on Windows is typically `npx.cmd` (batch file)
- Direct execution fails because Node.js can't directly execute batch files
- Wrapping with `cmd /c` launches Command Prompt to run the batch file

**What was happening**:
```
Error: spawn npx ENOENT
```

**What's happening now**:
```bash
cmd.exe -> npx.cmd -> node.exe -> claude-flow
```

### MCP Context Size Issue

**Current Context Usage**:
- **flow-nexus**: 94 tools (~59,052 tokens) - 55% of total
- **ruv-swarm**: 25 tools (~15,935 tokens) - 15% of total
- **playwright**: 21 tools (~13,678 tokens) - 13% of total
- **filesystem**: 14 tools (~9,221 tokens) - 9% of total
- **agentic-payments**: 10 tools (~6,651 tokens) - 6% of total
- **Others**: ~2,300 tokens - 2% of total

**Total**: ~106,818 tokens (recommended limit: 25,000)

**Impact**:
- Higher token usage per conversation
- Slower initial loading
- May hit context limits faster

**Solutions**:
1. **Disable unused servers**: If you're not using flow-nexus (59k tokens), disable it
2. **Use selective enabling**: Enable servers only when needed via `@mention`
3. **Split by project**: Use different MCP configurations for different projects

---

## Next Steps

### Immediate (Do Now)

- [ ] Exit and restart Claude Code
- [ ] Run `/mcp` command to check server status
- [ ] Verify claude-flow connects successfully
- [ ] Verify ruv-swarm connects successfully
- [ ] Verify flow-nexus connects successfully

### Short Term (Within 1 Day)

- [ ] Decide on connascence-analyzer solution (disable/fix/remove)
- [ ] Test connascence CLI commands directly if needed
- [ ] Optimize CLAUDE.md file size
- [ ] Review which MCP servers are actually needed

### Long Term (Optional)

- [ ] Create project-specific .mcp.json files
- [ ] Document MCP server usage patterns
- [ ] Set up MCP server monitoring/health checks
- [ ] Consider creating wrapper scripts for frequently used MCP commands

---

## Technical Details

### File Locations

**Project MCP Config** (Shared):
```
C:\Users\17175\.mcp.json
```

**User MCP Config** (Global):
```
C:\Users\17175\.claude.json
```

**MCP Server Logs**:
```
C:\Users\17175\AppData\Local\claude-cli-nodejs\Cache\C--Users-17175
```

### Command for Debugging

**View MCP logs in real-time**:
```bash
claude --debug
```

**Check specific server logs**:
```bash
# Navigate to log directory
cd "C:\Users\17175\AppData\Local\claude-cli-nodejs\Cache\C--Users-17175"

# View latest logs
dir /OD  # List by date, newest last
type <latest-log-file>
```

---

## Verification Checklist

After restarting Claude Code, verify:

- [ ] No Windows wrapper warnings in `/mcp` output
- [ ] claude-flow shows "connected" status
- [ ] ruv-swarm shows "connected" status
- [ ] flow-nexus shows "connected" status
- [ ] Can successfully use at least one tool from each server
- [ ] No timeout errors in MCP server list
- [ ] Context size warnings reduced (if MCP servers disabled)

---

## Contact & Support

If issues persist:

1. **Check MCP docs**: https://docs.claude.com/en/docs/claude-code/mcp
2. **View debug logs**: `claude --debug`
3. **Test servers individually**: Use commands in "Testing Required" section
4. **Check package versions**:
   ```bash
   cmd /c npx claude-flow@alpha --version
   cmd /c npx ruv-swarm --version
   cmd /c npx flow-nexus@latest --version
   ```

---

**Last Updated**: 2025-11-08
**Next Review**: After Claude Code restart and testing

