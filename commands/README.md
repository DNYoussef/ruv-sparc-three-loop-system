# Command Reference - RUV SPARC Three-Loop System

**Total Commands**: 224 (149 original + 75 new)
**MECE Coverage**: 100% across 10 domains
**Documentation**: See [MASTER-COMMAND-INDEX.md](../docs/MASTER-COMMAND-INDEX.md)

---

## 📋 Quick Navigation

All 224 commands are organized into 10 MECE (Mutually Exclusive, Collectively Exhaustive) domains:

1. **Core Infrastructure** (3 commands) - System help, memory, swarm
2. **Agent Lifecycle** (18 commands) - Agent spawning, coordination, health monitoring
3. **Development Workflow** (67 commands) - SPARC methodology, workflows, feature development
4. **Quality & Validation** (29 commands) - Audits, testing, benchmarking
5. **Performance & Optimization** (18 commands) - Profiling, resource optimization
6. **Memory & State** (18 commands) - Memory management, checkpointing
7. **Monitoring & Telemetry** (18 commands) - Observability, metrics, tracing
8. **Integration & External** (21 commands) - GitHub, deployment, CI/CD
9. **Research & Analysis** (20 commands) - Reverse engineering, research workflows
10. **Automation & Hooks** (22 commands) - Lifecycle hooks, automation

---

## 📚 Complete Documentation

**Master Index**: [MASTER-COMMAND-INDEX.md](../docs/MASTER-COMMAND-INDEX.md)
- Complete command reference with descriptions
- Organized by MECE domain
- Cross-references and command chains
- Usage patterns and integration examples

**MECE Taxonomy**: [MECE-COMMAND-TAXONOMY.md](../docs/command-taxonomy/MECE-COMMAND-TAXONOMY.md)
- Complete gap analysis
- Pattern identification
- Coverage metrics
- Implementation phases

**GraphViz Workflows**: [docs/workflows/graphviz/](../docs/workflows/graphviz/)
- 25 workflow diagrams
- Visual documentation for all major workflows
- Deployment pipelines, testing pyramids, CI/CD flows

---

## 🚀 Quick Start Examples

### Feature Development
```bash
/build-feature "user authentication" → /regression-test → /e2e-test → /github-release
```

### Complete CI/CD
```bash
/workflow:cicd && /security-audit && /dependency-audit && /e2e-test && /load-test && /workflow:deployment && /monitoring-configure
```

### Performance Optimization
```bash
/profiler-start && /resource-optimize && /memory-optimize && /cpu-optimize && /network-optimize && /bundle-optimize && /profiler-stop
```

### Research Pipeline
```bash
/research:literature-review → /research:experiment-design → /research:data-analysis → /research:paper-write → /research:citation-manager
```

---

## 📁 Directory Structure

```
commands/
├── agent-commands/         # Agent lifecycle (18 commands)
├── audit-commands/         # Quality & validation (29 commands)
├── essential-commands/     # Core features (build, deploy, fix, review, check)
├── github/                 # GitHub integration (21 commands)
├── hooks/automation/       # Automation & hooks (22 commands)
├── memory/                 # Memory & state (18 commands)
├── monitoring/             # Monitoring & telemetry (18 commands)
├── optimization/           # Performance & optimization (18 commands)
├── re/                     # Reverse engineering (7 commands)
├── research/               # Research workflows (13 commands)
├── sparc/                  # SPARC methodology (31 commands)
├── swarm/                  # Swarm coordination (18 commands)
└── workflows/              # Development workflows (13 commands)
```

---

## ✨ New Commands (Phase 1-4)

### Phase 1: Critical Infrastructure (21 commands)
- `/security-audit` - Comprehensive security scanning
- `/dependency-audit` - Dependency vulnerabilities
- `/docker-build` - Docker containerization
- `/docker-deploy` - Docker deployment
- `/k8s-deploy` - Kubernetes deployment
- `/github-release` - Automated releases
- `/workflow:cicd` - CI/CD orchestration
- `/workflow:deployment` - Deployment workflow
- `/workflow:rollback` - Rollback workflow
- `/regression-test` - Regression testing
- `/integration-test` - Integration testing
- `/e2e-test` - End-to-end testing
- `/load-test` - Load testing
- `/smoke-test` - Smoke tests
- `/agent-health-check` - Agent monitoring
- `/monitoring-configure` - Monitoring setup
- `/alert-configure` - Alert thresholds
- `/log-stream` - Log streaming
- `/trace-request` - Distributed tracing
- `/profiler-start` - Performance profiling
- `/profiler-stop` - Stop profiler

### Phase 2: Memory & State (14 commands)
- `/memory-clear` - Clear memory
- `/memory-export` - Export snapshots
- `/memory-import` - Import snapshots
- `/memory-merge` - Merge memories
- `/memory-stats` - Usage statistics
- `/memory-gc` - Garbage collection
- `/state-checkpoint` - Checkpointing
- `/state-restore` - State restoration
- `/state-diff` - State comparison
- `/agent-retire` - Agent retirement
- `/agent-clone` - Clone agents
- `/agent-benchmark` - Benchmark agents
- `/coordination-visualize` - Visualize topology
- `/metrics-export` - Export metrics

### Phase 3: Development Enhancement (20 commands)
- `/sparc:api-designer` - API design
- `/sparc:database-architect` - DB design
- `/sparc:frontend-specialist` - Frontend expert
- `/sparc:backend-specialist` - Backend expert
- `/sparc:mobile-specialist` - Mobile expert
- `/workflow:testing` - Testing workflow
- `/workflow:hotfix` - Hotfix workflow
- `/license-audit` - License compliance
- `/accessibility-audit` - A11y compliance
- `/performance-benchmark` - Benchmarking
- `/resource-optimize` - Resource optimization
- `/memory-optimize` - Memory optimization
- `/cpu-optimize` - CPU optimization
- `/network-optimize` - Network optimization
- `/bundle-optimize` - Bundle optimization
- `/query-optimize` - Query optimization
- `/render-optimize` - Render optimization
- `/cloudflare-deploy` - Cloudflare deployment
- `/vercel-deploy` - Vercel deployment

### Phase 4: Integration & Research (26 commands)
- `/aws-deploy` - AWS deployment
- `/github-actions` - GitHub Actions
- `/github-pages` - GitHub Pages
- `/jira-sync` - Jira integration
- `/slack-notify` - Slack notifications
- `/docker-compose` - Docker Compose
- `/terraform-apply` - Terraform IaC
- `/ansible-deploy` - Ansible deployment
- `/re:malware-sandbox` - Malware sandboxing
- `/re:network-traffic` - Network analysis
- `/re:memory-dump` - Memory dump analysis
- `/re:decompile` - Decompilation
- `/research:literature-review` - Literature review
- `/research:experiment-design` - Experiment design
- `/research:data-analysis` - Data analysis
- `/research:paper-write` - Paper writing
- `/research:citation-manager` - Citations
- `/hook:on-error` - Error hooks
- `/hook:on-success` - Success hooks
- `/hook:on-commit` - Commit hooks
- `/hook:on-push` - Push hooks
- `/hook:on-pr` - PR hooks
- `/hook:on-deploy` - Deploy hooks
- `/automation:retry-failed` - Retry logic
- `/automation:schedule-task` - Scheduling
- `/automation:cron-job` - Cron jobs

---

## 🎯 Command Categories

### By Frequency of Use

**Daily Use** (Essential):
- `/sparc`, `/build-feature`, `/fix-bug`, `/quick-check`, `/review-pr`

**Weekly Use** (Quality):
- `/audit-pipeline`, `/performance-benchmark`, `/security-audit`, `/deploy-check`

**As Needed** (Specialized):
- `/re:quick`, `/research:literature-review`, `/swarm-init`, `/github-release`

### By User Type

**Individual Developers**:
- Core SPARC commands (`/sparc`, `/sparc:coder`, `/sparc:debug`)
- Quality checks (`/quick-check`, `/theater-detect`)
- Bug fixing (`/fix-bug`)

**Team Leads**:
- Review commands (`/review-pr`, `/code-review`)
- Workflows (`/workflow:development`, `/workflow:testing`)
- Deployment (`/deploy-check`, `/github-release`)

**DevOps Engineers**:
- Deployment (`/docker-deploy`, `/k8s-deploy`, `/aws-deploy`)
- Monitoring (`/monitoring-configure`, `/alert-configure`, `/log-stream`)
- CI/CD (`/workflow:cicd`, `/github-actions`)

**Security Engineers**:
- Audits (`/security-audit`, `/dependency-audit`, `/license-audit`)
- Compliance (`/accessibility-audit`)

**Researchers**:
- Research workflows (`/research:*`)
- Reverse engineering (`/re:*`)
- Analysis (`/assess-risks`, `/prisma-init`)

---

## 🔗 Related Documentation

- **Master Command Index**: Complete reference with examples and cross-links
- **MECE Taxonomy**: Gap analysis and coverage metrics
- **GraphViz Diagrams**: Visual workflow documentation
- **Memory MCP Update**: Integration with persistent memory system

---

**Last Updated**: 2025-11-01
**Total Commands**: 224
**MECE Coverage**: 100%
