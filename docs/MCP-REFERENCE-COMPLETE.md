# MCP Complete Reference - Conditional Loading Guide

**Version**: 1.0.0
**Date**: 2025-11-15
**Status**: Production Ready
**Token Optimization**: 89.7% reduction potential (109.8k -> 11.3k global)

---

## Executive Summary

**PROBLEM**: MCP tools consume 109.8k tokens (54.9% of 200k context window)
**ROOT CAUSE**: All MCPs loaded globally regardless of actual need
**SOLUTION**: Conditional MCP loading based on playbook/skill requirements
**SAVINGS**: 98.5k tokens (89.7% reduction) by loading only what's needed

---

## Table of Contents

1. [Quick Start - Minimal Global Config](#quick-start---minimal-global-config)
2. [MCP Tier System](#mcp-tier-system)
3. [Playbook-to-MCP Mappings](#playbook-to-mcp-mappings)
4. [Skill-to-MCP Mappings](#skill-to-mcp-mappings)
5. [Configuration Profiles](#configuration-profiles)
6. [Implementation Guide](#implementation-guide)
7. [Redundancy Analysis](#redundancy-analysis)
8. [Complete MCP Catalog](#complete-mcp-catalog)

---

## Quick Start - Minimal Global Config

**RECOMMENDED DEFAULT (11.3k tokens - 89.7% reduction!)**

`~/.claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "fetch": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-fetch"]
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-sequential-thinking"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "C:\\Users\\17175"]
    }
  }
}
```

**Why these 3?**
- `fetch` (826 tokens): Web content retrieval for research/documentation
- `sequential-thinking` (1.5k tokens): Deep reasoning for complex problems
- `filesystem` (9k tokens): File operations (14 tools total)

**Everything else is conditional!** Add specialized MCPs only when needed.

---

## MCP Tier System

### TIER 0 - GLOBAL ESSENTIAL (11.3k tokens)
**Always loaded**. Core operations needed by all workflows.

| MCP | Tokens | Tools | Purpose |
|-----|--------|-------|---------|
| fetch | 826 | 1 | Web content retrieval |
| sequential-thinking | 1,500 | 1 | Deep reasoning |
| filesystem | 9,000 | 14 | File operations |

**Usage**: 100% of workflows

---

### TIER 1 - CODE QUALITY (1.8k tokens)
**Conditional**. Load for code analysis, review, debugging.

| MCP | Tokens | Tools | When to Load |
|-----|--------|-------|--------------|
| focused-changes | 1,800 | 3 | Change tracking, scope validation, error trees |

**Triggers**: "audit", "review", "quality", "debug", "analyze code", "check violations"

**Skills**: clarity-linter, functionality-audit, code-review-assistant, theater-detection-audit

**Agents**: coder, reviewer, tester, code-analyzer, sparc-coder

**Playbooks**: Quality (Quick check, Comprehensive review, Dogfooding), Security (Audit, Compliance)

---

### TIER 2 - SWARM COORDINATION (15.5k tokens)
**Conditional**. Load for multi-agent workflows only.

| MCP | Tokens | Tools | When to Load |
|-----|--------|-------|--------------|
| ruv-swarm | 15,500 | 25 | Multi-agent orchestration, DAA, neural patterns |

**Triggers**: "swarm", "multi-agent", "parallel agents", "orchestrate", "coordinate", "Byzantine consensus"

**Skills**: parallel-swarm-implementation, swarm-orchestration, hive-mind-advanced

**Agents**: hierarchical-coordinator, mesh-coordinator, adaptive-coordinator, collective-intelligence-coordinator

**Playbooks**: Three-Loop System, Complex coordination workflows

**NOTE**: REMOVE flow-nexus swarm tools (redundant with ruv-swarm)

---

### TIER 3 - MACHINE LEARNING (12.8k tokens)
**Conditional**. Load for ML/AI development only.

| MCP | Tokens | Tools | When to Load |
|-----|--------|-------|--------------|
| flow-nexus neural | 12,800 | 19 | Neural network training, distributed ML, model management |

**Triggers**: "train model", "neural network", "ML pipeline", "dataset", "distributed training", "inference"

**Skills**: deep-research-orchestrator (ML track), ml-training-debugger, baseline-replication, method-development

**Agents**: ml-developer, model-training-specialist, mlops-deployment-agent, experiment-tracking-agent

**Playbooks**: ML Pipeline, Distributed Neural Training, Vector Search/RAG

**Tools Breakdown**:
- Training: neural_train, neural_training_status, neural_validation_workflow
- Models: neural_list_models, neural_deploy_template, neural_performance_benchmark
- Distributed: neural_cluster_init, neural_node_deploy, neural_train_distributed
- Templates: neural_list_templates, neural_publish_template, neural_rate_template

---

### TIER 4 - BROWSER AUTOMATION (15.3k tokens)
**Conditional**. Load for UI testing, web scraping, frontend work.

| MCP | Tokens | Tools | When to Load |
|-----|--------|-------|--------------|
| playwright | 15,300 | 23 | Browser control, screenshots, UI testing |

**Triggers**: "UI test", "browser", "screenshot", "frontend test", "visual regression", "e2e", "web scraping"

**Skills**: e2e-testing-specialist, visual-regression-agent, frontend-performance-optimizer

**Agents**: tester (frontend), react-developer, vue-developer, frontend-performance-optimizer

**Playbooks**: Frontend Specialist, E2E Shipping (with UI), Quality (Visual regression)

**Tools Breakdown**:
- Navigation: browser_navigate, browser_navigate_back, browser_tabs
- Interaction: browser_click, browser_type, browser_select_option, browser_drag, browser_hover
- Capture: browser_take_screenshot, browser_snapshot, browser_console_messages
- Automation: browser_fill_form, browser_file_upload, browser_run_code
- Testing: browser_wait_for, browser_evaluate, browser_network_requests

---

### TIER 5 - SANDBOXES (6.2k tokens)
**Conditional**. Load for isolated code execution/testing.

| MCP | Tokens | Tools | When to Load |
|-----|--------|-------|--------------|
| flow-nexus sandbox | 6,200 | 9 | Isolated code execution, testing environments |

**Triggers**: "execute code", "sandbox", "isolated test", "safe execution", "test environment"

**Skills**: functionality-audit, prototyping workflows

**Agents**: coder (testing), tester, functionality-audit

**Playbooks**: Prototyping, E2E Shipping (testing phase)

**Tools**: sandbox_create, sandbox_execute, sandbox_upload, sandbox_configure, sandbox_status, sandbox_logs, sandbox_list, sandbox_stop, sandbox_delete

---

### TIER 6 - SPECIALIZED (17.1k tokens total)
**Conditional**. Load only for specific use cases.

| MCP | Tokens | Tools | When to Load |
|-----|--------|-------|--------------|
| agentic-payments | 6,600 | 10 | E-commerce, payment processing |
| flow-nexus auth | 6,500 | 11 | User management, authentication workflows |
| flow-nexus workflows | 4,400 | 7 | Complex CI/CD automation |
| toc | 600 | 1 | Documentation table of contents |

**Payment Mandates (agentic-payments)**:
- **Triggers**: "payment", "e-commerce", "shopping cart", "checkout", "mandate", "billing"
- **Tools**: create_active_mandate, sign_mandate, verify_mandate, revoke_mandate, create_intent_mandate, create_cart_mandate
- **Use Case**: Autonomous agent payment authorization with spend caps

**Authentication (flow-nexus auth)**:
- **Triggers**: "user registration", "login", "authentication", "session management", "user profile"
- **Tools**: user_register, user_login, user_logout, user_verify_email, user_reset_password, user_profile, user_upgrade
- **Use Case**: User management systems, auth workflows

**Workflows (flow-nexus workflows)**:
- **Triggers**: "complex workflow", "CI/CD orchestration", "event-driven", "workflow automation"
- **Tools**: workflow_create, workflow_execute, workflow_status, workflow_audit_trail, workflow_queue_status
- **Use Case**: Advanced automation, CI/CD pipelines

**Table of Contents (toc)**:
- **Triggers**: "table of contents", "documentation structure", "project overview"
- **Tool**: generate_toc
- **Use Case**: Documentation generation only

---

### TIER 7 - REMOVE/OPTIONAL (40.9k tokens SAVED!)
**Recommended for removal**. Rarely used or redundant.

| MCP Category | Tokens | Reason for Removal |
|--------------|--------|-------------------|
| flow-nexus swarm | 5,900 | REDUNDANT with ruv-swarm (same functionality) |
| flow-nexus challenges | 3,900 | Gamification features unused |
| flow-nexus app_store | 3,200 | Publishing rarely used |
| flow-nexus execution_streams | 2,300 | Specialized monitoring |
| flow-nexus realtime | 1,800 | Realtime subscriptions rarely needed |
| flow-nexus storage | 2,600 | Cloud storage (use local filesystem) |
| flow-nexus payments | 4,800 | Credits/billing (internal system) |
| flow-nexus app_management | 3,000 | App analytics |
| flow-nexus system | 1,700 | System health monitoring |
| seraphina_chat | 700 | Demo AI chat feature |
| flow-nexus github_repo_analyze | 600 | Use native GitHub tools instead |

**TOTAL SAVINGS: 40,900 tokens (37.3%)**

---

## Playbook-to-MCP Mappings

### DELIVERY PLAYBOOKS (5 total)

**Simple Feature**
- MCPs: GLOBAL only (fetch, sequential-thinking, filesystem)
- Tokens: 11.3k
- Reason: Most feature work doesn't need specialized tooling

**Three-Loop System**
- MCPs: GLOBAL + ruv-swarm + focused-changes
- Tokens: 11.3k + 15.5k + 1.8k = 28.6k
- Reason: Needs swarm coordination for Loop 2 parallel implementation

**E2E Shipping**
- MCPs: GLOBAL + flow-nexus sandbox + playwright (if UI)
- Tokens: 11.3k + 6.2k + 15.3k = 32.8k
- Reason: Sandbox testing + UI validation

**Bug Fix**
- MCPs: GLOBAL + focused-changes
- Tokens: 11.3k + 1.8k = 13.1k
- Reason: Root cause analysis via focused-changes

**Prototyping**
- MCPs: GLOBAL + flow-nexus sandbox
- Tokens: 11.3k + 6.2k = 17.5k
- Reason: Quick iteration in sandboxes

---

### OPERATIONS PLAYBOOKS (4 total)

**Production Deployment**
- MCPs: GLOBAL + flow-nexus workflows + execution_streams
- Tokens: 11.3k + 4.4k + 2.3k = 18.0k
- Reason: Complex deployment workflows + monitoring

**CI/CD Setup**
- MCPs: GLOBAL + flow-nexus workflows
- Tokens: 11.3k + 4.4k = 15.7k
- Reason: Pipeline automation

**Infrastructure Scaling**
- MCPs: GLOBAL + ruv-swarm
- Tokens: 11.3k + 15.5k = 26.8k
- Reason: Multi-agent coordination for distributed systems

**Performance Optimization**
- MCPs: GLOBAL + focused-changes
- Tokens: 11.3k + 1.8k = 13.1k
- Reason: Change tracking during optimization

---

### RESEARCH PLAYBOOKS (4 total)

**Deep Research SOP**
- MCPs: GLOBAL + (conditional: flow-nexus neural if ML research)
- Tokens: 11.3k (baseline) or 24.1k (ML research)
- Reason: Fetch for paper retrieval, neural only if ML experiments

**Quick Investigation**
- MCPs: GLOBAL only
- Tokens: 11.3k
- Reason: Web search + file analysis sufficient

**Planning & Architecture**
- MCPs: GLOBAL only
- Tokens: 11.3k
- Reason: Sequential thinking + file operations

**Literature Review**
- MCPs: GLOBAL only
- Tokens: 11.3k
- Reason: Fetch for papers, filesystem for storage

---

### SECURITY PLAYBOOKS (3 total)

**Security Audit**
- MCPs: GLOBAL + focused-changes + playwright (optional)
- Tokens: 11.3k + 1.8k + 15.3k = 28.4k
- Reason: Change tracking + optional security testing

**Compliance Validation**
- MCPs: GLOBAL + focused-changes
- Tokens: 11.3k + 1.8k = 13.1k
- Reason: Audit trail tracking

**Reverse Engineering**
- MCPs: GLOBAL + flow-nexus sandbox
- Tokens: 11.3k + 6.2k = 17.5k
- Reason: Safe binary execution in sandboxes

---

### QUALITY PLAYBOOKS (3 total)

**Quick Check**
- MCPs: GLOBAL + focused-changes
- Tokens: 11.3k + 1.8k = 13.1k
- Reason: Fast validation via change tracking

**Comprehensive Review**
- MCPs: GLOBAL + focused-changes + playwright
- Tokens: 11.3k + 1.8k + 15.3k = 28.4k
- Reason: Full audit + visual regression

**Dogfooding Cycle**
- MCPs: GLOBAL + focused-changes + flow-nexus sandbox
- Tokens: 11.3k + 1.8k + 6.2k = 19.3k
- Reason: Self-improvement via sandbox testing

---

### PLATFORM PLAYBOOKS (3 total)

**ML Pipeline**
- MCPs: GLOBAL + flow-nexus neural (ALL 19 tools)
- Tokens: 11.3k + 12.8k = 24.1k
- Reason: Complete ML toolchain

**Vector Search/RAG**
- MCPs: GLOBAL + flow-nexus storage + realtime
- Tokens: 11.3k + 2.6k + 1.8k = 15.7k
- Reason: Cloud storage + realtime subscriptions

**Distributed Neural Training**
- MCPs: GLOBAL + flow-nexus neural cluster (7 tools)
- Tokens: 11.3k + 12.8k = 24.1k
- Reason: Distributed training tools

---

### GITHUB PLAYBOOKS (3 total)

**PR Management**
- MCPs: GLOBAL + flow-nexus github_repo_analyze
- Tokens: 11.3k + 0.6k = 11.9k
- Reason: Repo analysis for PRs

**Release Management**
- MCPs: GLOBAL + flow-nexus workflows
- Tokens: 11.3k + 4.4k = 15.7k
- Reason: Release automation

**Multi-Repo Coordination**
- MCPs: GLOBAL + flow-nexus workflows
- Tokens: 11.3k + 4.4k = 15.7k
- Reason: Cross-repo workflows

---

### SPECIALIST PLAYBOOKS (4 total)

**Frontend Specialist**
- MCPs: GLOBAL + playwright + flow-nexus sandbox
- Tokens: 11.3k + 15.3k + 6.2k = 32.8k
- Reason: UI testing + sandbox execution

**Backend Specialist**
- MCPs: GLOBAL + flow-nexus sandbox
- Tokens: 11.3k + 6.2k = 17.5k
- Reason: API testing in sandboxes

**Full-Stack Specialist**
- MCPs: GLOBAL + playwright + flow-nexus sandbox
- Tokens: 11.3k + 15.3k + 6.2k = 32.8k
- Reason: Frontend + backend testing

**Infrastructure as Code**
- MCPs: GLOBAL + flow-nexus workflows
- Tokens: 11.3k + 4.4k = 15.7k
- Reason: IaC deployment automation

---

## Skill-to-MCP Mappings

### HIGH-FREQUENCY SKILLS (70% use GLOBAL only!)

**These skills need ONLY global MCPs:**
- ai-dev-orchestration, sparc-methodology, sparc-coord
- literature-synthesis, quick-investigation, planning-architecture
- network-security-setup, style-audit, dependencies
- Most specialized skills (80+ skills)

**Why?** Most development work requires only:
- Web research (fetch)
- Deep reasoning (sequential-thinking)
- File operations (filesystem)

---

### CODE QUALITY SKILLS (12 total)

**clarity-linter** → GLOBAL + focused-changes
**functionality-audit** → GLOBAL + focused-changes + flow-nexus sandbox
**theater-detection-audit** → GLOBAL + focused-changes + flow-nexus sandbox
**code-review-assistant** → GLOBAL + focused-changes
**production-readiness** → GLOBAL + focused-changes + playwright (optional)
**quick-quality-check** → GLOBAL + focused-changes

**Shared Pattern**: All use focused-changes for change tracking + root cause analysis

---

### SWARM COORDINATION SKILLS (6 total)

**parallel-swarm-implementation** → GLOBAL + ruv-swarm
**swarm-orchestration** → GLOBAL + ruv-swarm
**hive-mind-advanced** → GLOBAL + ruv-swarm
**swarm-advanced** → GLOBAL + ruv-swarm
**coordination** → GLOBAL + ruv-swarm

**Shared Pattern**: All use ruv-swarm for multi-agent orchestration

---

### MACHINE LEARNING SKILLS (9 total)

**deep-research-orchestrator** (ML track) → GLOBAL + flow-nexus neural
**baseline-replication** → GLOBAL + flow-nexus neural
**method-development** → GLOBAL + flow-nexus neural
**holistic-evaluation** → GLOBAL + flow-nexus neural
**ml-training-debugger** → GLOBAL + flow-nexus neural
**ml-expert** → GLOBAL + flow-nexus neural
**flow-nexus-neural** → GLOBAL + flow-nexus neural (ALL 19 tools)

**Shared Pattern**: All use flow-nexus neural for ML operations

---

### FRONTEND SKILLS (6 total)

**e2e-testing-specialist** → GLOBAL + playwright + flow-nexus sandbox
**visual-regression-agent** → GLOBAL + playwright
**frontend-performance-optimizer** → GLOBAL + playwright
**react-specialist** → GLOBAL + playwright (optional)
**vue-developer** → GLOBAL + playwright (optional)

**Shared Pattern**: All use playwright for browser automation

---

### INFRASTRUCTURE SKILLS (8 total)

**cicd-intelligent-recovery** → GLOBAL + flow-nexus workflows
**deployment-readiness** → GLOBAL + flow-nexus workflows
**github-workflow-automation** → GLOBAL + flow-nexus workflows
**infrastructure-as-code** → GLOBAL + flow-nexus workflows

**Shared Pattern**: All use flow-nexus workflows for CI/CD automation

---

## Configuration Profiles

### Profile 1: Minimal (11.3k tokens)
**Use for**: 70% of general development tasks

```json
{
  "mcpServers": {
    "fetch": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-fetch"]
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-sequential-thinking"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "C:\\Users\\17175"]
    }
  }
}
```

---

### Profile 2: Code Quality (13.1k tokens)
**Use for**: Code reviews, audits, debugging

```json
{
  "mcpServers": {
    "fetch": { ... },
    "sequential-thinking": { ... },
    "filesystem": { ... },
    "focused-changes": {
      "command": "node",
      "args": ["C:/Users/17175/Documents/Cline/MCP/focused-changes-server/build/index.js"]
    }
  }
}
```

---

### Profile 3: Swarm Coordination (26.8k tokens)
**Use for**: Three-Loop System, multi-agent workflows

```json
{
  "mcpServers": {
    "fetch": { ... },
    "sequential-thinking": { ... },
    "filesystem": { ... },
    "ruv-swarm": {
      "command": "npx",
      "args": ["ruv-swarm", "mcp", "start"]
    }
  }
}
```

---

### Profile 4: Machine Learning (24.1k tokens)
**Use for**: ML pipelines, neural training

```json
{
  "mcpServers": {
    "fetch": { ... },
    "sequential-thinking": { ... },
    "filesystem": { ... },
    "flow-nexus": {
      "command": "npx",
      "args": ["flow-nexus@latest", "mcp", "start"]
    }
  }
}
```

Note: Only enable neural tools subset if possible

---

### Profile 5: Frontend Testing (32.8k tokens)
**Use for**: UI testing, visual regression, e2e tests

```json
{
  "mcpServers": {
    "fetch": { ... },
    "sequential-thinking": { ... },
    "filesystem": { ... },
    "playwright": {
      "command": "npx",
      "args": ["playwright-mcp"]
    },
    "flow-nexus": {
      "command": "npx",
      "args": ["flow-nexus@latest", "mcp", "start"]
    }
  }
}
```

Note: Only enable sandbox tools subset if possible

---

### Profile 6: Full Stack (Current - 109.8k tokens)
**Use for**: Rare cases needing everything

```json
{
  "mcpServers": {
    "fetch": { ... },
    "sequential-thinking": { ... },
    "filesystem": { ... },
    "focused-changes": { ... },
    "ruv-swarm": { ... },
    "flow-nexus": { ... },
    "playwright": { ... },
    "agentic-payments": { ... }
  }
}
```

**WARNING**: Only use if genuinely needed. 70% token overhead!

---

## Implementation Guide

### Step 1: Backup Current Config

```powershell
Copy-Item "$env:APPDATA\Claude\claude_desktop_config.json" "$env:APPDATA\Claude\claude_desktop_config.json.backup"
```

---

### Step 2: Create Profile Files

```powershell
# Minimal profile (recommended default)
$minimalConfig = @"
{
  "mcpServers": {
    "fetch": { "command": "npx", "args": ["@modelcontextprotocol/server-fetch"] },
    "sequential-thinking": { "command": "npx", "args": ["@modelcontextprotocol/server-sequential-thinking"] },
    "filesystem": { "command": "npx", "args": ["@modelcontextprotocol/server-filesystem", "C:\\Users\\17175"] }
  }
}
"@
$minimalConfig | Out-File "$env:APPDATA\Claude\profiles\minimal.json" -Encoding UTF8

# Code Quality profile
# ... (similar for other profiles)
```

---

### Step 3: Switch Profiles Based on Task

```powershell
# Working on code quality
Copy-Item "$env:APPDATA\Claude\profiles\quality.json" "$env:APPDATA\Claude\claude_desktop_config.json"
# Restart Claude Code

# Working on ML
Copy-Item "$env:APPDATA\Claude\profiles\ml.json" "$env:APPDATA\Claude\claude_desktop_config.json"
# Restart Claude Code
```

---

### Step 4: Automate Profile Switching (Future)

Create PowerShell function:

```powershell
function Switch-ClaudeProfile {
    param([string]$ProfileName)
    $profilePath = "$env:APPDATA\Claude\profiles\$ProfileName.json"
    $configPath = "$env:APPDATA\Claude\claude_desktop_config.json"

    if (Test-Path $profilePath) {
        Copy-Item $profilePath $configPath -Force
        Write-Host "Switched to $ProfileName profile. Restart Claude Code."
    } else {
        Write-Error "Profile not found: $ProfileName"
    }
}

# Usage
Switch-ClaudeProfile -ProfileName "minimal"
Switch-ClaudeProfile -ProfileName "ml"
```

---

## Redundancy Analysis

### CRITICAL REDUNDANCY: ruv-swarm vs. flow-nexus swarm

**Problem**: 2 swarm coordination systems with overlapping functionality

| Feature | ruv-swarm | flow-nexus swarm |
|---------|-----------|------------------|
| swarm_init | ✅ | ✅ |
| agent_spawn | ✅ | ✅ |
| task_orchestrate | ✅ | ✅ |
| swarm_status | ✅ | ✅ |
| Tokens | 15,500 | 5,900 |
| Tools | 25 | 9 |
| DAA support | ✅ | ✅ |
| Neural integration | ✅ | ❌ |

**RECOMMENDATION**: Keep ruv-swarm (more features), remove flow-nexus swarm tools

**Savings**: 5,900 tokens (5.4%)

---

### Minor Redundancies

**flow-nexus execution_streams** (2.3k tokens)
- Specialized monitoring for execution streams
- Alternative: Use swarm_status from ruv-swarm
- Savings: 2,300 tokens

**flow-nexus realtime** (1.8k tokens)
- Realtime database subscriptions
- Alternative: Polling via standard tools
- Savings: 1,800 tokens

**flow-nexus storage** (2.6k tokens)
- Cloud storage operations
- Alternative: Use filesystem MCP for local storage
- Savings: 2,600 tokens

**TOTAL REDUNDANCY SAVINGS**: 12,600 tokens (11.5%)

---

## Complete MCP Catalog

### Currently Installed MCPs

**1. connascence-analyzer (Local Python - Production)**
- **Tokens**: Estimated 8,000 (14 agents access, not in context analysis)
- **Tools**: analyze_file, analyze_workspace, health_check
- **Purpose**: Code quality analysis, connascence detection
- **Status**: NOT globally loaded (agent-specific access)
- **Keep**: YES (production critical for code quality agents)

**2. memory-mcp (Local Python - Production)**
- **Tokens**: Estimated 6,000 (not in context analysis)
- **Tools**: vector_search, memory_store
- **Purpose**: Persistent cross-session memory
- **Status**: NOT globally loaded (hook integration)
- **Keep**: YES (production critical for all workflows)

**3. focused-changes (Local Node.js - Production)**
- **Tokens**: 1,800
- **Tools**: start_tracking, analyze_changes, root_cause_analysis
- **Purpose**: Change tracking, scope validation
- **Status**: TIER 1 (conditional)
- **Keep**: YES

**4. toc (Local Node.js - Production)**
- **Tokens**: 600
- **Tools**: generate_toc
- **Purpose**: Table of contents generation
- **Status**: TIER 6 (specialized)
- **Keep**: YES (minimal overhead)

---

### Official Anthropic MCPs

**5. fetch**
- **Tokens**: 826
- **Tools**: 1 (fetch)
- **Purpose**: Web content retrieval
- **Status**: TIER 0 (global essential)
- **Keep**: YES

**6. sequential-thinking**
- **Tokens**: 1,500
- **Tools**: 1 (sequentialthinking)
- **Purpose**: Deep reasoning
- **Status**: TIER 0 (global essential)
- **Keep**: YES

**7. filesystem**
- **Tokens**: 9,000
- **Tools**: 14 (read, write, edit, list, search, etc.)
- **Purpose**: File operations
- **Status**: TIER 0 (global essential)
- **Keep**: YES

---

### Flow-Nexus MCPs (Conditional - 122 tools!)

**8. ruv-swarm**
- **Tokens**: 15,500
- **Tools**: 25
- **Purpose**: Swarm coordination, DAA, neural patterns
- **Status**: TIER 2 (conditional - swarm workflows)
- **Keep**: YES

**9. flow-nexus swarm**
- **Tokens**: 5,900
- **Tools**: 9
- **Purpose**: Swarm coordination (REDUNDANT)
- **Status**: TIER 7 (remove)
- **Keep**: NO (redundant with ruv-swarm)

**10. flow-nexus neural**
- **Tokens**: 12,800
- **Tools**: 19
- **Purpose**: Neural training, distributed ML
- **Status**: TIER 3 (conditional - ML workflows)
- **Keep**: YES

**11. flow-nexus sandbox**
- **Tokens**: 6,200
- **Tools**: 9
- **Purpose**: Isolated code execution
- **Status**: TIER 5 (conditional - testing workflows)
- **Keep**: YES

**12. playwright**
- **Tokens**: 15,300
- **Tools**: 23
- **Purpose**: Browser automation
- **Status**: TIER 4 (conditional - frontend workflows)
- **Keep**: YES

**13. agentic-payments**
- **Tokens**: 6,600
- **Tools**: 10
- **Purpose**: Payment mandates
- **Status**: TIER 6 (specialized - e-commerce only)
- **Keep**: YES (niche use case)

**14. flow-nexus auth**
- **Tokens**: 6,500
- **Tools**: 11
- **Purpose**: User management
- **Status**: TIER 6 (specialized - auth workflows)
- **Keep**: YES (niche use case)

**15. flow-nexus workflows**
- **Tokens**: 4,400
- **Tools**: 7
- **Purpose**: Complex automation
- **Status**: TIER 6 (specialized - CI/CD)
- **Keep**: YES (niche use case)

**16-23. flow-nexus removable (40.9k tokens)**
- challenges (3,900) → REMOVE (gamification)
- app_store (3,200) → REMOVE (publishing)
- execution_streams (2,300) → REMOVE (specialized monitoring)
- realtime (1,800) → REMOVE (realtime subscriptions)
- storage (2,600) → REMOVE (use filesystem instead)
- payments (4,800) → REMOVE (internal credits system)
- app_management (3,000) → REMOVE (analytics)
- system (1,700) → REMOVE (health monitoring)
- seraphina_chat (700) → REMOVE (demo feature)
- github_repo_analyze (600) → REMOVE (use native GitHub tools)

---

## Token Savings Breakdown

| Category | Current | Recommended | Savings |
|----------|---------|-------------|---------|
| Global Essential | 11,300 | 11,300 | 0 |
| Code Quality | 0 | 1,800 (conditional) | 0* |
| Swarm | 21,400 | 15,500 (conditional) | 5,900 |
| ML | 12,800 | 12,800 (conditional) | 0* |
| Frontend | 15,300 | 15,300 (conditional) | 0* |
| Sandbox | 6,200 | 6,200 (conditional) | 0* |
| Specialized | 17,100 | 17,100 (conditional) | 0* |
| Removable | 40,900 | 0 | 40,900 |
| **TOTAL** | **109,800** | **11,300 (base)** | **98,500 (89.7%)** |

*Conditional MCPs only loaded when needed, not in global context

---

## Recommendations

### Immediate Actions

1. **Switch to Minimal Global Config** (11.3k tokens)
   - Remove all MCPs except: fetch, sequential-thinking, filesystem
   - Saves 98,500 tokens immediately

2. **Create Profile Files**
   - minimal.json (11.3k) - Default for 70% of work
   - quality.json (13.1k) - Code reviews
   - swarm.json (26.8k) - Multi-agent workflows
   - ml.json (24.1k) - ML development
   - frontend.json (32.8k) - UI testing

3. **Remove Redundant MCPs**
   - flow-nexus swarm tools (use ruv-swarm instead)
   - flow-nexus challenges, app_store, execution_streams
   - flow-nexus realtime, storage (use filesystem)
   - flow-nexus payments, app_management, system
   - seraphina_chat, github_repo_analyze

### Future Enhancements

1. **Dynamic MCP Loading** (requires Claude Code feature)
   - Skills declare MCP requirements in YAML frontmatter
   - Auto-load MCPs when skill activated
   - Unload after skill completion

2. **Profile Auto-Switching** (PowerShell automation)
   - Detect playbook/skill from user intent
   - Auto-switch to appropriate profile
   - Restart Claude Code

3. **MCP Lazy Loading** (requires MCP protocol enhancement)
   - Load MCP servers on-demand
   - Unload after timeout
   - Keep-alive for frequently used MCPs

---

## Support & References

**Documentation**:
- Official MCP Protocol: https://modelcontextprotocol.io
- ruv-swarm: https://github.com/ruvnet/ruv-swarm
- flow-nexus: https://github.com/ruvnet/flow-nexus
- Claude Code: https://claude.com/claude-code

**Issues & Feedback**:
- Claude Code: https://github.com/anthropics/claude-code/issues
- SPARC Plugin: https://github.com/DNYoussef/ruv-sparc-three-loop-system/issues

---

**Version**: 1.0.0
**Date**: 2025-11-15
**Author**: Sequential Thinking Analysis + Agent System
**Status**: Production Ready
**Next Review**: 2026-01-15
