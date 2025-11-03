# Skills Directory

**Status:** 121 skills organised into ten functional categories  
**Last reorganised:** 2025-11-02  
**Structure baseline:** Every skill retains the shared template (`skill.md`, `README.md`, `examples/`, `references/`, `resources/`, `tests/`).

---

## Quick Category Overview

| Category | Description | Skill count | Example skills |
|----------|-------------|-------------|----------------|
| delivery | Build and implementation workflows that ship product features | 8 | `delivery/feature-dev-complete`, `delivery/smart-bug-fix` |
| foundry | Tooling for creating new agents, skills, and prompts | 11 | `foundry/agent-creator`, `foundry/prompt-architect` |
| operations | DevOps, infrastructure, and release management capabilities | 18 | `operations/cloud-platforms`, `operations/production-readiness` |
| orchestration | Multi-agent coordination, swarms, and workflow execution | 15 | `orchestration/advanced-coordination`, `orchestration/swarm-orchestration` |
| platforms | Core platform components and intelligent data services | 13 | `platforms/agentdb`, `platforms/flow-nexus-platform` |
| quality | Testing, audit, review, and continuous verification skills | 21 | `quality/code-review-assistant`, `quality/quick-quality-check` |
| research | Research pipelines, analysis, and discovery processes | 10 | `research/deep-research-orchestrator`, `research/literature-synthesis` |
| security | Security, compliance, and reverse engineering expertise | 10 | `security/network-security-setup`, `security/reverse-engineering-deep` |
| specialists | Domain specialists for languages, stacks, and ML | 9 | `specialists/language-specialists`, `specialists/ml-expert` |
| tooling | Supporting utilities and cross-channel adapters | 6 | `tooling/web-cli-teleport`, `tooling/pptx-generation` |

Totals include nested skills; for example `operations/cloud-platforms/aws-specialist` counts toward the operations total.

---

## Directory Layout

```
skills/
  delivery/                  # Build and implementation workflows
  foundry/                   # Skill and prompt creation toolchain
  operations/                # DevOps, infrastructure, release automation
  orchestration/             # Swarms, cascades, and workflow control
  platforms/                 # AgentDB, Flow Nexus, ReasoningBank, ML
  quality/                   # Audits, testing SOPs, code review pipelines
  research/                  # Discovery, replication, and analysis tracks
  security/                  # Compliance, sandboxing, reverse engineering
  specialists/               # Domain experts and specialist stacks
  tooling/                   # Utilities, presentation, CLI bridges
  references/                # Master index and shared reference docs
  _pipeline-automation/      # Enhancement scripts and audit reports
```

Within each category, skill folders keep their original structure and metadata. Family groupings (for example `cloud-platforms/*` or `machine-learning/*`) stay intact beneath their category.

---

## Standard Skill Structure

Every skill follows the same MECE-aligned layout:

```
skill-name/
  skill.md          # YAML front matter + imperative playbook
  README.md         # Quick start guide and reference
  examples/         # Worked examples (basic → advanced)
  references/       # Best practices, API notes, troubleshooting
  resources/        # Templates, scripts, assets (optional)
  tests/            # Validation and scenario coverage (optional)
```

Silver-tier and higher skills include at least the README, three examples, two reference documents, and a graphviz workflow.

---

## Finding The Right Skill

1. **Browse by category** – start in the relevant top-level directory and scan the folder names.  
2. **Use the master index** – `skills/references/master-skills-index.md` lists every skill with tags and trigger phrases.  
3. **Search by feature** – `rg "skill-name"` from the repository root will locate the skill definition, SOPs, and usage.

Related SOPs remain beside their functional area. Example: code review procedures live in `quality/sop-code-review`, while infrastructure automations are under `operations/`.

---

## Supporting Material

- `references/master-skills-index.md` – canonical cross reference of all skills and tags.  
- `references/master-skills-index.md#catalogue` – direct links into every skill folder after the reorganisation.  
- `_pipeline-automation/` – scripts used for bulk enhancement, auditing, and MECE compliance checks.  
- Category `_shared` folders – hold auxiliary examples or resources that apply to an entire family (for example `operations/observability/_shared`).

---

## Contributing Updates

1. Place new skills inside the category that matches their primary function.  
2. Follow the standard folder structure and Silver-tier baseline.  
3. Update the category table above and the taxonomy index (`docs/skills-taxonomy/INDEX.md`) when adding or moving skills.  
4. Run the pipeline audits in `_pipeline-automation` if you introduce large batches of changes.

This organisation is intended to make navigation and discovery faster while preserving the detailed content that already existed for every skill.
