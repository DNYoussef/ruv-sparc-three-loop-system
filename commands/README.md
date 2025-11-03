# Commands Directory

**Status:** 219 slash commands organised into ten functional categories  
**Last reorganised:** 2025-11-02  
**Command format:** Each command file starts with YAML front matter (`name`, `description`, `category`, `dependencies`, etc.) followed by usage, examples, and chaining guidance.

---

## Quick Category Overview

| Category | Description | Command count | Example commands |
|----------|-------------|---------------|------------------|
| delivery | SPARC execution, development workflows, training pipelines, and day-to-day build/test automation | 67 | `delivery/essential-commands/build-feature.md`, `delivery/workflows/workflow-cicd.md`, `delivery/sparc/sparc.md` |
| foundry | Agent creation and lifecycle utilities, registry maintenance, and meta-command templates | 9 | `foundry/agent-commands/agent-benchmark.md`, `foundry/agents/agent-list.md`, `foundry/templates/github-pr-manager.md` |
| operations | CI/CD, deployment, automation hooks, monitoring, memory management, and optimisation utilities | 74 | `operations/hooks/hook-on-deploy.md`, `operations/monitoring/metrics-export.md`, `operations/optimization/resource-optimize.md` |
| orchestration | Swarm coordination, hive-mind operations, and command-level workflow orchestration | 26 | `orchestration/swam/swarm-init.md`, `orchestration/hive-mind/queen-orchestrate.md`, `orchestration/coordination/orchestrate.md` |
| platforms | Flow Nexus and multi-model integrations across Codex, Gemini, and other external services | 4 | `platforms/multi-model-commands/gemini-media.md`, `platforms/multi-model-commands/codex-auto.md` |
| quality | Audits, benchmarking, phase reports, testing orchestration, and validation utilities | 16 | `quality/audit-commands/security-audit.md`, `quality/analysis/performance-report.md`, `quality/reports/PHASE4_COMPLETION_SUMMARY.md` |
| research | Research automation, experiment pipelines, and analysis helpers | 9 | `research/research:data-analysis.md`, `research/research:literature-review.md`, `research/research:paper-write.md` |
| security | Reverse engineering and security-focused command set (static, dynamic, malware analysis) | 11 | `security/re/decompile.md`, `security/re/network-traffic.md`, `security/re/dynamic.md` |
| specialists | (Reserved) Domain-specific command packs for future specialist tooling | 0 | – |
| tooling | Command reference helpers, CLAUDE Flow quick guides, and supporting documentation | 3 | `tooling/claude-flow/claude-flow-help.md`, `tooling/claude-flow/claude-flow-memory.md` |

Totals include nested directories; for example `operations/monitoring/*` contributes to the operations total.

---

## Directory Layout

```
commands/
  delivery/               # SPARC, workflows, essential day-to-day commands
  foundry/                # Agent lifecycle, registry tooling, command templates
  operations/             # Automation hooks, monitoring, CI/CD, optimisation, memory
  orchestration/          # Swarm, hive-mind, coordination utilities
  platforms/              # Flow Nexus and multi-model integrations
  quality/                # Audits, benchmarking, reports, testing orchestrators
  research/               # Research and analysis pipelines
  security/               # Reverse engineering and security tooling
  specialists/            # Reserved for domain-specific command packs
  tooling/                # Reference material and helper docs
```

Within each category the legacy domain structure is preserved (for example `delivery/essential-commands/*.md`).

---

## Command File Structure

Every command uses the same structure:

```
---
name: workflow:cicd
description: End-to-end CI/CD orchestration
category: delivery
dependencies: [build-feature, regression-test, deploy-check]
chains_with: [security-audit, github-release]
---

# /workflow:cicd

## Overview
...
```

The front matter captures metadata (dependencies, chains, tags) while the body documents usage patterns, examples, and automation notes.

---

## Finding the Right Command

1. **Browse by category** – open the folder matching your workflow (for example `operations/monitoring/` for observability commands).  
2. **Use the master index** – see `docs/MASTER-COMMAND-INDEX.md` for a fully searchable reference (update forthcoming to match this structure).  
3. **Search by capability** – `rg "/workflow" commands/delivery` or `rg "memory" commands/operations` to locate specific commands.  
4. **Cross-reference agents/skills** – the reorganised folders mirror `agents/` and `skills/`, making it easier to align command packs with agent families.

---

## Updating or Adding Commands

1. Place new commands in the category that reflects their primary function.  
2. Follow the YAML + documentation structure above, documenting dependencies and compatible chains.  
3. Update this README and `docs/command-taxonomy/INDEX.md` with new counts and paths.  
4. Regenerate any CLI help or registry artefacts that enumerate all commands.

This unified layout keeps commands, agents, and skills aligned by functional area, simplifying discovery and maintenance across the Three-Loop system.
