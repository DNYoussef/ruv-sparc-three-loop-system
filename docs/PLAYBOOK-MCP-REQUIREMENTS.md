# Playbook MCP Requirements - Complete Reference

**Version**: 1.0.0
**Date**: 2025-11-15
**Status**: Production Ready
**Skills Analyzed**: 137
**Playbooks Documented**: 29

---

## Overview

This document maps all 29 playbooks to their MCP requirements. Since playbooks are **skill sequences**, their MCP needs are inherited from constituent skills. All skills now have integrated MCP documentation - this guide aggregates those requirements at the playbook level.

## Quick Reference

| Playbook Category | Typical MCPs | Token Cost | Use Case |
|-------------------|--------------|------------|----------|
| **Simple Delivery** | None (built-in tools) | 11.3k (global only) | Basic features |
| **Complex Delivery** | flow-nexus, memory-mcp | 44.9k (22.5%) | Multi-week projects |
| **Code Quality** | connascence, focused-changes, memory-mcp | 31.7k (15.9%) | Audits, reviews |
| **SPARC/Swarm** | claude-flow, ruv-swarm | 27.8k (13.9%) | Multi-agent workflows |
| **Deep Research** | memory-mcp (REQUIRED) | 23.7k (11.9%) | Multi-month research |
| **Cloud/ML** | flow-nexus | 43.8k (21.9%) | Neural training, sandboxes |
| **GitHub** | flow-nexus (optional) | 11.3k-43.8k | Repository automation |

---

## Playbook Breakdown

### 1. Delivery Playbooks (5 playbooks)

#### 1.1 Simple Feature Implementation

**Skills Used**: sparc-methodology, quick-quality-check, functionality-audit, sop-code-review
**Time**: 2-4 hours
**MCP Requirements**: **NONE** (built-in tools only)

**Rationale**: Simple features use local development tools. No specialized MCPs needed.

**Token Cost**: 11.3k (global TIER 0 only)

---

#### 1.2 Complex Feature (Three-Loop) 🔥 FLAGSHIP

**Skills Used**:
- Loop 1: research-driven-planning, gemini-search
- Loop 2: parallel-swarm-implementation, theater-detection-audit
- Loop 3: cicd-intelligent-recovery

**Time**: 8-14 hours
**MCP Requirements**:
- `flow-nexus` (32.5k) - Gemini search, sandbox testing, parallel coordination
- `memory-mcp` (12.4k) - Cross-loop state persistence

**Activation**:
```powershell
claude mcp add flow-nexus npx flow-nexus@latest mcp start
claude mcp add memory-mcp  # Manual activation (see CONDITIONAL-MCP-ACTIVATION-GUIDE.md)
```

**Token Cost**: 56.2k (28.1% of context)
**Use When**: Building production-grade features requiring research, multi-agent implementation, automated testing

---

#### 1.3 End-to-End Feature Shipping

**Skills Used**: feature-dev-complete (12-stage workflow)
**Time**: 6-12 hours
**MCP Requirements**:
- `flow-nexus` (32.5k) - Gemini search, architecture design, deployment
- `memory-mcp` (12.4k) - Stage progression tracking

**Token Cost**: 56.2k (28.1%)

---

#### 1.4 Quick Bug Fix

**Skills Used**: smart-bug-fix
**Time**: 30-90 minutes
**MCP Requirements**:
- `flow-nexus` (32.5k) - Codex sandbox iteration
- `memory-mcp` (12.4k) - Bug pattern learning

**Token Cost**: 56.2k (28.1%)
**Alternative**: Skip MCPs for simple bugs, use built-in debugging

---

#### 1.5 Rapid Prototyping

**Skills Used**: codex-auto, sandbox-configurator
**Time**: 1-3 hours
**MCP Requirements**:
- `flow-nexus` (32.5k) - Codex autonomous execution in sandboxes

**Token Cost**: 43.8k (21.9%)

---

### 2. Operations Playbooks (4 playbooks)

#### 2.1 Production Deployment

**Skills Used**: deployment-readiness, production-readiness
**Time**: 4-8 hours
**MCP Requirements**:
- `flow-nexus` (32.5k) - Production benchmarks, A/B testing validation
- `memory-mcp` (12.4k) - Deployment history, rollback patterns

**Token Cost**: 56.2k (28.1%)

---

#### 2.2 CI/CD Pipeline Setup

**Skills Used**: cicd-intelligent-recovery, github-workflow-automation
**Time**: 2-6 hours
**MCP Requirements**:
- `flow-nexus` (32.5k) - Sandbox testing, workflow validation
- `memory-mcp` (12.4k) - Failure pattern learning

**Token Cost**: 56.2k (28.1%)

---

#### 2.3 Infrastructure Scaling

**Skills Used**: kubernetes-specialist, terraform-iac, aws-specialist
**Time**: 4-12 hours
**MCP Requirements**:
- `flow-nexus` (32.5k) - Infrastructure testing in sandboxes
- `memory-mcp` (12.4k) - Configuration versioning

**Token Cost**: 56.2k (28.1%)

---

#### 2.4 Performance Optimization

**Skills Used**: performance-analysis, perf-analyzer
**Time**: 2-6 hours
**MCP Requirements**: **NONE** (local profiling tools)

**Token Cost**: 11.3k (global only)

---

### 3. Research Playbooks (4 playbooks)

#### 3.1 Deep Research SOP (Multi-Month) 🔥 FLAGSHIP

**Skills Used**: deep-research-orchestrator, literature-synthesis, baseline-replication, method-development, holistic-evaluation, gate-validation
**Time**: 1-6 months
**MCP Requirements** (CRITICAL):
- `memory-mcp` (12.4k) - **REQUIRED** for multi-month state persistence
- `flow-nexus` (32.5k) - Neural training, distributed experiments

**Activation**:
```powershell
# REQUIRED before starting deep research
claude mcp add memory-mcp  # Manual activation
claude mcp add flow-nexus npx flow-nexus@latest mcp start  # For ML experiments
```

**Token Cost**: 56.2k (28.1%)
**Critical**: Memory MCP is **non-negotiable** for Deep Research SOP. Without it, state is lost between sessions.

---

#### 3.2 Quick Investigation

**Skills Used**: researcher, gemini-search
**Time**: 30-120 minutes
**MCP Requirements**:
- `flow-nexus` (32.5k) - Gemini grounded search (optional)

**Token Cost**: 11.3k (global only, fetch MCP sufficient)
**Alternative**: Use built-in fetch MCP for basic research

---

#### 3.3 Planning & Architecture

**Skills Used**: research-driven-planning, system-architect
**Time**: 2-4 hours
**MCP Requirements**:
- `memory-mcp` (12.4k) - Pre-mortem cycle persistence
- `flow-nexus` (32.5k) - Gemini search for best practices

**Token Cost**: 56.2k (28.1%)

---

#### 3.4 Literature Review

**Skills Used**: literature-synthesis
**Time**: 4-8 hours
**MCP Requirements**:
- `fetch` (0.826k) - Already global, for ArXiv/Semantic Scholar access
- `memory-mcp` (12.4k) - Citation management, synthesis tracking

**Token Cost**: 23.7k (11.9%)

---

### 4. Security Playbooks (3 playbooks)

#### 4.1 Security Audit

**Skills Used**: security testing agents, sop-code-review
**Time**: 4-8 hours
**MCP Requirements**:
- `connascence-analyzer` (5.1k) - Code security scanning
- `focused-changes` (1.8k) - Change scope validation
- `memory-mcp` (12.4k) - Security pattern learning

**Token Cost**: 31.7k (15.9%)

---

#### 4.2 Compliance Validation

**Skills Used**: wcag-accessibility, compliance agents
**Time**: 2-6 hours
**MCP Requirements**:
- `playwright` (4.2k) - Accessibility automation
- `memory-mcp` (12.4k) - Compliance checklist tracking

**Token Cost**: 27.9k (14.0%)

---

#### 4.3 Reverse Engineering (Authorized/CTF)

**Skills Used**: reverse-engineering-quick, reverse-engineering-deep
**Time**: 2-8 hours
**MCP Requirements**:
- `flow-nexus` (32.5k) - Isolated sandbox for binary analysis
- `memory-mcp` (12.4k) - Malware pattern database

**Token Cost**: 56.2k (28.1%)
**Security**: Always execute in sandboxed environment

---

### 5. Quality Playbooks (3 playbooks)

#### 5.1 Quick Quality Check

**Skills Used**: quick-quality-check
**Time**: 5-15 minutes
**MCP Requirements**: **NONE** (parallel lint/test/security via Bash)

**Token Cost**: 11.3k (global only)

---

#### 5.2 Comprehensive Review

**Skills Used**: code-review-assistant, clarity-linter, functionality-audit
**Time**: 1-4 hours
**MCP Requirements**:
- `connascence-analyzer` (5.1k) - Coupling detection
- `focused-changes` (1.8k) - Change tracking
- `memory-mcp` (12.4k) - Review pattern learning

**Token Cost**: 31.7k (15.9%)

---

#### 5.3 Dogfooding Cycle (Self-Improvement)

**Skills Used**: sop-dogfooding-quality-detection, sop-dogfooding-pattern-retrieval, sop-dogfooding-continuous-improvement
**Time**: 2-4 minutes per cycle
**MCP Requirements**:
- `connascence-analyzer` (5.1k) - Violation detection
- `focused-changes` (1.8k) - Fix scope validation
- `memory-mcp` (12.4k) - **REQUIRED** for pattern retrieval

**Token Cost**: 31.7k (15.9%)
**Automation**: Can run every 30-60 minutes for continuous improvement

---

### 6. Platform Playbooks (3 playbooks)

#### 6.1 ML Pipeline Development

**Skills Used**: ml-expert, ml-developer, method-development
**Time**: 1-4 days
**MCP Requirements**:
- `flow-nexus` (32.5k) - Distributed neural training
- `memory-mcp` (12.4k) - Experiment tracking

**Token Cost**: 56.2k (28.1%)

---

#### 6.2 Vector Search/RAG System

**Skills Used**: agentdb-vector-search, agentdb-optimization
**Time**: 2-6 hours
**MCP Requirements**: **NONE** (AgentDB is npm package, not MCP)

**Token Cost**: 11.3k (global only)
**Note**: AgentDB integration via npm, not MCP server

---

#### 6.3 Distributed Neural Training

**Skills Used**: flow-nexus-neural
**Time**: 4-12 hours
**MCP Requirements**:
- `flow-nexus` (32.5k) - **REQUIRED** for E2B sandbox cluster

**Token Cost**: 43.8k (21.9%)

---

### 7. GitHub Playbooks (3 playbooks)

#### 7.1 PR Management & Review

**Skills Used**: github-code-review, code-review-assistant
**Time**: 30-120 minutes
**MCP Requirements**:
- `connascence-analyzer` (5.1k) - Code quality checks
- `memory-mcp` (12.4k) - Review pattern learning
- `flow-nexus` (32.5k) - Optional GitHub integration

**Token Cost**: 31.7k (15.9%) without flow-nexus, 63.9k (32.0%) with

---

#### 7.2 Release Management

**Skills Used**: github-release-management
**Time**: 1-3 hours
**MCP Requirements**:
- `flow-nexus` (32.5k) - Optional automated release coordination
- Built-in GitHub CLI sufficient for basic releases

**Token Cost**: 11.3k (global) to 43.8k (with flow-nexus)

---

#### 7.3 Multi-Repo Coordination

**Skills Used**: github-multi-repo
**Time**: 2-6 hours
**MCP Requirements**:
- `flow-nexus` (32.5k) - Optional cross-repo coordination
- `memory-mcp` (12.4k) - Dependency tracking

**Token Cost**: 56.2k (28.1%)

---

### 8. Specialist Playbooks (4 playbooks)

#### 8.1 Frontend Development (React/Vue)

**Skills Used**: react-specialist, vue-developer, frontend-performance-optimizer
**Time**: 4-12 hours
**MCP Requirements**: **NONE** (npm tooling sufficient)

**Token Cost**: 11.3k (global only)

---

#### 8.2 Backend API Development

**Skills Used**: sop-api-development, backend-dev
**Time**: 1-2 weeks
**MCP Requirements**:
- `memory-mcp` (12.4k) - Multi-week state persistence
- `ruv-swarm` (15.5k) - Multi-agent coordination

**Token Cost**: 39.2k (19.6%)

---

#### 8.3 Full-Stack Application

**Skills Used**: feature-dev-complete, react-specialist, backend-dev
**Time**: 2-4 weeks
**MCP Requirements**:
- `flow-nexus` (32.5k) - End-to-end testing, deployment
- `memory-mcp` (12.4k) - Project state persistence

**Token Cost**: 56.2k (28.1%)

---

#### 8.4 Infrastructure as Code

**Skills Used**: terraform-iac, docker-containerization, kubernetes-specialist
**Time**: 1-3 days
**MCP Requirements**:
- `flow-nexus` (32.5k) - Infrastructure testing in sandboxes
- `memory-mcp` (12.4k) - Configuration versioning

**Token Cost**: 56.2k (28.1%)

---

## Activation Patterns by Use Case

### Pattern 1: Code Quality Focus (31.7k tokens)
```powershell
claude mcp add connascence-analyzer  # Manual activation
claude mcp add focused-changes       # Manual activation
claude mcp add memory-mcp            # Manual activation
```
**Use For**: Code reviews, audits, dogfooding, security scanning

---

### Pattern 2: Research/Planning Focus (56.2k tokens)
```powershell
claude mcp add flow-nexus npx flow-nexus@latest mcp start
claude mcp add memory-mcp  # Manual activation
```
**Use For**: Deep research, architecture planning, multi-week projects

---

### Pattern 3: ML/Cloud Focus (56.2k tokens)
```powershell
claude mcp add flow-nexus npx flow-nexus@latest mcp start
claude mcp add memory-mcp  # Manual activation
```
**Use For**: Neural training, distributed experiments, cloud deployment

---

### Pattern 4: Swarm Coordination Focus (39.2k tokens)
```powershell
claude mcp add claude-flow npx claude-flow@alpha mcp start
claude mcp add ruv-swarm npx ruv-swarm mcp start
claude mcp add memory-mcp  # Manual activation
```
**Use For**: Multi-agent workflows, SPARC methodology, complex coordination

---

### Pattern 5: Minimal (11.3k tokens) - DEFAULT
No additional MCPs needed!
**Use For**: Simple features, basic development, quick tasks

---

## Token Budget Planning

| Project Type | Recommended MCPs | Token Cost | % of 200k Context |
|--------------|------------------|------------|-------------------|
| **Quick Task** | None (global only) | 11.3k | 5.7% |
| **Standard Feature** | flow-nexus OR memory-mcp | 23.7k-43.8k | 11.9-21.9% |
| **Complex Project** | flow-nexus + memory-mcp | 56.2k | 28.1% |
| **Code Quality** | connascence + focused + memory | 31.7k | 15.9% |
| **Full Stack** | All needed MCPs | 56.2k-70k | 28-35% |

**Rule of Thumb**: Reserve 50-60% of context for actual work, limit MCPs to 30-40% maximum.

---

## Best Practices

1. **Start Minimal**: Begin with global MCPs only (11.3k)
2. **Activate on Demand**: Add MCPs when playbook explicitly requires them
3. **Monitor Token Usage**: Each playbook documents its MCP costs
4. **Deactivate After**: Remove MCPs when project phase completes
5. **Use Automation Scripts**: Create project-specific activation profiles (see CONDITIONAL-MCP-ACTIVATION-GUIDE.md)
6. **Plan Ahead**: Know your playbook's MCP needs before starting

---

## Summary

**Total Playbooks**: 29
**Require No MCPs**: 8 playbooks (27.6%) - Simple tasks, local tooling sufficient
**Require 1-2 MCPs**: 13 playbooks (44.8%) - Standard complexity
**Require 3+ MCPs**: 8 playbooks (27.6%) - Complex/long-term projects

**Key Insight**: 72.4% of playbooks need specialized MCPs, but only when actively using those playbooks. Default state uses just 11.3k tokens (5.7% context).

**Deep Research & Three-Loop are the only "MCP-heavy" playbooks** - everything else uses 11.3k-56.2k tokens conditionally.

For complete activation instructions, see: `C:\Users\17175\docs\CONDITIONAL-MCP-ACTIVATION-GUIDE.md`