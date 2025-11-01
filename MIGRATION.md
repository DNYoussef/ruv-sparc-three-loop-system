# Migration Guide: v2.x → v3.0 (Official Claude Code Plugins)

**Date**: November 1, 2025
**Target**: Existing users of ruv-sparc-three-loop-system v2.x
**Goal**: Migrate to official Claude Code plugin marketplace (v3.0)

---

## 🎯 What's Changed

### v2.x (Old Structure)
```
ruv-sparc-three-loop-system/
├── .claude/
│   ├── skills/
│   ├── commands/
│   └── agents/
├── hooks/
├── security/
└── docs/
```

**Installation**: Manual repository cloning and setup
**Usage**: All skills/agents/commands loaded at once
**Updates**: Manual git pull required

---

### v3.0 (New Plugin Structure)
```
ruv-sparc-three-loop-system/
├── .claude-plugin/
│   ├── marketplace.json
│   └── README.md
├── plugins/
│   ├── 12fa-core/
│   ├── 12fa-three-loop/
│   ├── 12fa-security/
│   ├── 12fa-visual-docs/
│   └── 12fa-swarm/
├── .claude/         # Existing skills/agents/commands (preserved)
├── hooks/          # Existing hooks (preserved)
├── security/       # Existing security components (preserved)
└── docs/           # Existing documentation (preserved)
```

**Installation**: `/plugin marketplace add` + `/plugin install`
**Usage**: Modular - install only what you need
**Updates**: Automatic through Claude Code plugin system

---

## 🚀 Migration Steps

### Step 1: Backup Your Current Setup
```bash
# Backup your local repository
cd /path/to/ruv-sparc-three-loop-system
git status  # Check for uncommitted changes
git stash   # Stash any local changes

# Create backup branch (optional)
git checkout -b backup-v2
git checkout main
```

---

### Step 2: Pull Latest v3.0 Changes
```bash
# Fetch and pull latest changes
git fetch origin
git pull origin main

# Verify you have v3.0
cat .claude-plugin/marketplace.json | grep version
# Should output: "version": "3.0.0"
```

---

### Step 3: Remove Manual Claude Setup (if applicable)

If you previously added the repository to `.claude/config`:

```bash
# Remove old manual config
# Edit your ~/.config/claude/config.json
# Remove any manual paths to ruv-sparc-three-loop-system
```

---

### Step 4: Add Plugin Marketplace
```bash
# Inside Claude Code, run:
/plugin marketplace add DNYoussef/ruv-sparc-three-loop-system
```

**Expected Output**:
```
✅ Marketplace added successfully
📦 5 plugins available:
   - 12fa-core
   - 12fa-three-loop
   - 12fa-security
   - 12fa-visual-docs
   - 12fa-swarm

Use /plugin install <plugin-name> to install
```

---

### Step 5: Install Desired Plugins

#### Option A - Core Only (Recommended for Most Users)
```bash
/plugin install 12fa-core
```

This gives you:
- SPARC methodology
- 10 core skills
- 12 core agents
- 11 commands (`/sparc`, `/audit-pipeline`, `/quick-check`, etc.)
- Quality gates and hooks

---

#### Option B - Everything You Had in v2.x
```bash
/plugin install 12fa-core 12fa-three-loop 12fa-security 12fa-visual-docs 12fa-swarm
```

This gives you ALL features from v2.x:
- Complete SPARC workflow
- Three-Loop Architecture
- All 6 security components
- All 271 Graphviz diagrams
- All swarm coordination

---

### Step 6: Verify MCP Servers

Check that required MCP servers are configured:

```bash
# Inside Claude Code
/mcp list
```

**Expected**:
```
✅ claude-flow@alpha - Active
✅ ruv-swarm - Active (if using 12fa-swarm)
✅ flow-nexus - Active (optional)
```

**If Missing**, install them:

```bash
# Required (for 12fa-core and all plugins)
npm install -g claude-flow@alpha
claude mcp add claude-flow npx claude-flow@alpha mcp start

# Required for 12fa-swarm
npm install -g ruv-swarm
claude mcp add ruv-swarm npx ruv-swarm mcp start

# Optional (for advanced features)
npm install -g flow-nexus@latest
claude mcp add flow-nexus npx flow-nexus@latest mcp start
```

---

### Step 7: Test Installation

Run a simple command to verify everything works:

```bash
/sparc "Create a simple hello world function with tests"
```

**Expected Behavior**:
- SPARC phases execute (Specification → Pseudocode → Architecture → Refinement → Code)
- Tests are generated automatically
- Code is implemented with TDD

**If it works**: ✅ Migration successful!
**If errors**: See Troubleshooting below.

---

## 🔄 Functional Equivalents (v2.x → v3.0)

### Commands

| v2.x Command | v3.0 Plugin | v3.0 Command | Notes |
|--------------|-------------|--------------|-------|
| Manual SPARC execution | `12fa-core` | `/sparc` | Same functionality, official command |
| Manual audit pipeline | `12fa-core` | `/audit-pipeline` | Automated quality gates |
| Manual theater detection | `12fa-core` | `/theater-detect` | 6-agent Byzantine consensus |
| Manual Three-Loop | `12fa-three-loop` | `/development` | Automated research → implement → recover |
| Manual swarm init | `12fa-swarm` | `/swarm-init` | Official swarm topology command |
| Manual security review | `12fa-security` | `/sparc:security-review` | Comprehensive security audit |

---

### Skills

All v2.x skills are preserved in v3.0 plugins:

| v2.x Skill | v3.0 Plugin | Status |
|------------|-------------|--------|
| agent-creator | `12fa-core` | ✅ Included |
| sparc-methodology | `12fa-core` | ✅ Included |
| research-driven-planning | `12fa-three-loop` | ✅ Included |
| parallel-swarm-implementation | `12fa-three-loop` | ✅ Included |
| cicd-intelligent-recovery | `12fa-three-loop` | ✅ Included |
| network-security-setup | `12fa-security` | ✅ Included |
| swarm-advanced | `12fa-swarm` | ✅ Included |
| hive-mind-advanced | `12fa-swarm` | ✅ Included |

---

### Agents

All v2.x agents are preserved in v3.0 plugins:

| v2.x Agent | v3.0 Plugin | Status |
|------------|-------------|--------|
| coder | `12fa-core` | ✅ Included |
| reviewer | `12fa-core` | ✅ Included |
| tester | `12fa-core` | ✅ Included |
| planner | `12fa-core` | ✅ Included |
| researcher | `12fa-core` | ✅ Included |
| task-orchestrator | `12fa-three-loop` | ✅ Included |
| cicd-engineer | `12fa-three-loop` | ✅ Included |
| security-manager | `12fa-security` | ✅ Included |
| queen-coordinator | `12fa-swarm` | ✅ Included |
| byzantine-coordinator | `12fa-swarm` | ✅ Included |

---

## 🐛 Troubleshooting

### Issue: "Plugin marketplace not found"

**Solution**:
```bash
# Ensure you're using Claude Code >= 2.0.13
claude --version

# If outdated, update Claude Code
# Follow instructions at https://claude.com/code
```

---

### Issue: "MCP server 'claude-flow' not found"

**Solution**:
```bash
# Install and configure claude-flow MCP
npm install -g claude-flow@alpha
claude mcp add claude-flow npx claude-flow@alpha mcp start

# Verify
claude mcp list
```

---

### Issue: "/sparc command not found"

**Solution**:
```bash
# Verify 12fa-core is installed
/plugin list

# If missing, install it
/plugin install 12fa-core

# Restart Claude Code
```

---

### Issue: "Graphviz diagrams not rendering"

**Solution**:
```bash
# Install Graphviz (required for 12fa-visual-docs)

# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz

# Windows
choco install graphviz

# Verify installation
dot -V
```

---

### Issue: "Vault connection failed" (for 12fa-security)

**Solution**:
```bash
# Install HashiCorp Vault

# macOS
brew install vault

# Start Vault dev server
vault server -dev

# Configure plugin
/setup  # Runs security infrastructure setup
```

---

## 📊 What You Get with Migration

### Before (v2.x)
- ❌ Manual repository setup
- ❌ All features loaded at once (slow)
- ❌ Manual updates required
- ❌ No official plugin support
- ⚠️ Unclear component dependencies

### After (v3.0)
- ✅ Official plugin marketplace
- ✅ Modular installation (install what you need)
- ✅ Automatic updates via Claude Code
- ✅ Official Claude Code plugin support
- ✅ Clear component dependencies
- ✅ Better performance (only load what you use)
- ✅ 5 well-documented plugin packages

---

## 🔗 Component Mapping

### If You Used SPARC Only (v2.x)
**Install**: `12fa-core`

**What you get**:
- All SPARC phases (Specification, Pseudocode, Architecture, Refinement, Code)
- Theater detection
- Quality gates
- All core commands

---

### If You Used Three-Loop System (v2.x)
**Install**: `12fa-core` + `12fa-three-loop`

**What you get**:
- Loop 1: Research-driven planning
- Loop 2: Parallel swarm implementation
- Loop 3: CI/CD intelligent recovery
- All research and automation features

---

### If You Used Security Components (v2.x)
**Install**: `12fa-core` + `12fa-security`

**What you get**:
- All 6 security components (Agent Spec Gen, Policy DSL, Guardrails, Registry, Vault, Telemetry)
- Security review commands
- Vault integration
- OpenTelemetry monitoring

---

### If You Used Graphviz Diagrams (v2.x)
**Install**: `12fa-core` + `12fa-visual-docs`

**What you get**:
- All 271 Graphviz diagrams
- Validation scripts (Bash + PowerShell)
- Interactive HTML viewer
- Master catalog and templates

---

### If You Used Swarm Coordination (v2.x)
**Install**: `12fa-core` + `12fa-swarm`

**What you get**:
- All 4 topologies (Hierarchical, Mesh, Adaptive, Ring)
- All 3 consensus protocols (Byzantine, Raft, Gossip)
- Hive Mind coordination
- GitHub integration

---

## 🎯 Recommended Migration Paths

### Path 1: Minimal Migration (Core Only)
**Best for**: Individual developers who used SPARC methodology

```bash
# 1. Add marketplace
/plugin marketplace add DNYoussef/ruv-sparc-three-loop-system

# 2. Install core
/plugin install 12fa-core

# Done! Start using /sparc commands
```

**Time**: ~2 minutes
**Complexity**: Low

---

### Path 2: Standard Migration (Core + Three-Loop)
**Best for**: Teams who used research-driven planning and parallel implementation

```bash
# 1. Add marketplace
/plugin marketplace add DNYoussef/ruv-sparc-three-loop-system

# 2. Install plugins
/plugin install 12fa-core 12fa-three-loop

# 3. Verify MCP servers
/mcp list

# Done! Start using /development commands
```

**Time**: ~5 minutes
**Complexity**: Medium

---

### Path 3: Full Migration (Everything)
**Best for**: Enterprises who used all v2.x features

```bash
# 1. Add marketplace
/plugin marketplace add DNYoussef/ruv-sparc-three-loop-system

# 2. Install all plugins
/plugin install 12fa-core 12fa-three-loop 12fa-security 12fa-visual-docs 12fa-swarm

# 3. Verify MCP servers
/mcp list

# 4. Install external dependencies
brew install graphviz vault prometheus grafana  # macOS

# 5. Setup security infrastructure
/setup

# Done! All v2.x features available
```

**Time**: ~15 minutes
**Complexity**: High

---

## ✅ Post-Migration Checklist

After migration, verify these work:

- [ ] `/sparc "Create hello world"` - Core SPARC workflow
- [ ] `/quick-check` - Fast quality validation
- [ ] `/audit-pipeline` - Complete quality gates (if 12fa-core installed)
- [ ] `/development "Feature description"` - Three-Loop workflow (if 12fa-three-loop installed)
- [ ] `/sparc:security-review` - Security audit (if 12fa-security installed)
- [ ] `bash docs/12fa/graphviz/validate-all-diagrams.sh` - Diagram validation (if 12fa-visual-docs installed)
- [ ] `/swarm-init mesh` - Swarm coordination (if 12fa-swarm installed)

---

## 🤝 Support

**Issues during migration?**

1. **Check this guide first** - Most issues are covered above
2. **GitHub Issues**: [Report migration issues](https://github.com/DNYoussef/ruv-sparc-three-loop-system/issues)
3. **GitHub Discussions**: [Ask migration questions](https://github.com/DNYoussef/ruv-sparc-three-loop-system/discussions)

---

## 📚 Additional Resources

- [Main README](README.md) - Complete system overview
- [Marketplace README](.claude-plugin/README.md) - Plugin installation guide
- [12fa-core README](plugins/12fa-core/README.md) - Core system documentation
- [12fa-three-loop README](plugins/12fa-three-loop/README.md) - Three-Loop Architecture
- [12fa-security README](plugins/12fa-security/README.md) - Security components
- [12fa-visual-docs README](plugins/12fa-visual-docs/README.md) - Visual documentation
- [12fa-swarm README](plugins/12fa-swarm/README.md) - Swarm coordination

---

**Version**: 3.0.0
**Migration Author**: DNYoussef
**Last Updated**: November 1, 2025

**✅ Migration Complete! Enjoy the official Claude Code plugin experience!**
