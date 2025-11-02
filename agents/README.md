# Agents Directory - Organization Guide

**Status**: 24 domain-organized directories + 131 total agents
**Last Updated**: 2025-11-02

---

## Directory Structure

The agents directory is organized into **24 specialized domains** for efficient agent discovery and categorization:

\`\`\`
agents/
├── analysis/          # Analytical & audit agents (root-cause-analyzer, audit-pipeline-orchestrator)
├── architecture/      # System architecture & design agents
├── business/          # Business logic & domain agents
├── consensus/         # Consensus protocol agents (byzantine, raft, gossip)
├── core/              # Core fundamental agents (coder, reviewer, base-template-generator)
├── data/              # Data processing & transformation agents
├── database/          # Database specialists (design, optimization, migration)
├── development/       # General development agents (backend-dev, mobile-dev, ml-developer)
├── devops/            # DevOps & infrastructure agents (cicd-engineer)
├── documentation/     # Documentation generation & management agents
├── flow-nexus/        # Multi-model integration (Gemini, Codex, multi-model-orchestrator)
├── frontend/          # Frontend development specialists (React, Vue, UI components)
├── github/            # GitHub integration agents (pr-manager, issue-tracker, workflow-automation)
├── goal/              # Goal-oriented & planning agents
├── hive-mind/         # Collective intelligence & queen-led coordination
├── neural/            # Neural network & ML model agents
├── optimization/      # Performance optimization specialists
├── reasoning/         # Reasoning & decision-making agents
├── research/          # Research & analysis agents (researcher, data-steward, ethics-agent)
├── sparc/             # SPARC methodology agents (specification, architecture, refinement)
├── registry/          # Agent registry management (registry.json, MCP scripts)
└── README.md          # This file
\`\`\`

---

## Quick Reference: Specialist Agent Types

When using Claude Code's Task tool, ALWAYS specify one of 5 specialist types:

| Type | Use For | Agents |
|------|---------|--------|
| \`researcher\` | Analysis, investigation, requirements | researcher, data-steward, ethics-agent |
| \`coder\` | Implementation, feature development | coder, backend-dev, mobile-dev, ml-developer |
| \`analyst\` | Testing, review, quality assurance | reviewer, tester, code-analyzer |
| \`optimizer\` | Performance tuning, optimization | perf-analyzer, performance-benchmarker |
| \`coordinator\` | Multi-agent orchestration | hierarchical-coordinator, mesh-coordinator |

See **CLAUDE.md** for complete specialist agent selection guide.

---

## Resources

- **Agent Registry**: \`agents/registry/registry.json\` (131 agents)
- **Complete Guide**: See CLAUDE.md for specialist agent selection
- **Examples**: See \`skills/\` directory for skill-agent integration

