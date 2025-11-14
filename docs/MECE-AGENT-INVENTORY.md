# MECE Agent Inventory for SOP Workflows

**Principle**: Mutually Exclusive, Collectively Exhaustive
**Purpose**: Complete agent-command-tool mapping for rewrites

---

## Unique Agent Types Across All SOPs

### 1. Market Researcher
**Used In**: sop-product-launch
**Slash Commands**:
- Universal: /file-read, /file-write, /memory-store, /memory-retrieve
- Specialist: /market-analysis, /competitor-research, /customer-survey, /trend-analysis

**MCP Tools**:
- mcp__ruv-swarm__agent_spawn (for sub-researchers)
- mcp__claude-flow__memory_store (store research findings)
- mcp__flow-nexus__market_data (market statistics)
- mcp__ruv-swarm__neural_train (pattern learning from past research)

**Triggers**: When need market intelligence, competitive analysis, customer insights

---

### 2. Business Analyst
**Used In**: sop-product-launch
**Slash Commands**:
- Universal: /file-read, /memory-retrieve, /communicate-report
- Specialist: /swot-analysis, /business-model-canvas, /revenue-projection, /risk-assessment

**MCP Tools**:
- mcp__claude-flow__memory_retrieve (get market research)
- mcp__ruv-swarm__agent_metrics (analyze performance data)
- mcp__flow-nexus__app_analytics (app performance data)

**Triggers**: When need strategic analysis, business model validation, risk assessment

---

### 3. Product Manager
**Used In**: sop-product-launch, sop-api-development
**Slash Commands**:
- Universal: /file-write, /memory-store, /agent-delegate
- Specialist: /product-roadmap, /feature-prioritization, /user-story-creation, /requirements-gathering

**MCP Tools**:
- mcp__ruv-swarm__task_orchestrate (coordinate development)
- mcp__claude-flow__memory_store (store requirements)
- mcp__flow-nexus__workflow_create (create product workflows)

**Triggers**: When need product strategy, roadmap planning, requirements definition

---

### 4. Backend Developer
**Used In**: sop-product-launch, sop-api-development
**Slash Commands**:
- Universal: /file-read, /file-write, /file-edit, /git-commit, /test-run
- Specialist: /api-design, /db-migrate, /auth-setup, /api-endpoint-create, /middleware-create

**MCP Tools**:
- mcp__flow-nexus__sandbox_create (isolated dev environment)
- mcp__flow-nexus__sandbox_execute (run code)
- mcp__claude-flow__memory_retrieve (get API specs)
- mcp__ruv-swarm__agent_spawn (spawn helper agents)

**Triggers**: When need REST API, GraphQL API, server-side logic, database integration

---

### 5. Frontend Developer
**Used In**: sop-product-launch
**Slash Commands**:
- Universal: /file-read, /file-write, /git-commit, /test-run
- Specialist: /component-create, /state-management-setup, /ui-design, /api-integration

**MCP Tools**:
- mcp__flow-nexus__sandbox_create (React/Vue environment)
- mcp__flow-nexus__template_deploy (frontend template)
- mcp__claude-flow__memory_retrieve (get API specs)

**Triggers**: When need web UI, React/Vue components, client-side logic

---

### 6. Mobile Developer
**Used In**: sop-product-launch
**Slash Commands**:
- Universal: /file-read, /file-write, /git-commit, /test-run
- Specialist: /mobile-component-create, /native-bridge-setup, /platform-build, /app-store-submit

**MCP Tools**:
- mcp__flow-nexus__sandbox_create (React Native environment)
- mcp__flow-nexus__template_deploy (mobile template)
- mcp__claude-flow__memory_retrieve (get API specs)

**Triggers**: When need iOS app, Android app, React Native, cross-platform mobile

---

### 7. Database Architect
**Used In**: sop-product-launch, sop-api-development
**Slash Commands**:
- Universal: /file-write, /memory-store
- Specialist: /schema-design, /query-optimize, /index-create, /migration-generate

**MCP Tools**:
- mcp__flow-nexus__sandbox_execute (test queries)
- mcp__claude-flow__memory_store (store schema)
- mcp__ruv-swarm__performance_benchmarker (query performance)

**Triggers**: When need database schema, query optimization, data modeling

---

### 8. Security Specialist
**Used In**: sop-product-launch, sop-api-development
**Slash Commands**:
- Universal: /file-read, /test-run, /communicate-report
- Specialist: /security-audit, /vulnerability-scan, /penetration-test, /compliance-check

**MCP Tools**:
- mcp__flow-nexus__sandbox_execute (security testing)
- mcp__ruv-swarm__agent_metrics (security metrics)
- mcp__claude-flow__memory_store (audit results)

**Triggers**: When need security audit, vulnerability assessment, compliance validation

---

### 9. QA Engineer / Tester
**Used In**: sop-product-launch, sop-api-development, sop-code-review
**Slash Commands**:
- Universal: /test-run, /test-coverage, /test-validate, /communicate-report
- Specialist: /test-suite-create, /integration-test, /e2e-test, /performance-test

**MCP Tools**:
- mcp__flow-nexus__sandbox_create (test environment)
- mcp__flow-nexus__sandbox_execute (run tests)
- mcp__ruv-swarm__benchmark_run (performance benchmarks)
- mcp__claude-flow__memory_retrieve (test plan)

**Triggers**: When need test automation, quality validation, test coverage

---

### 10. System Architect
**Used In**: sop-api-development, sop-code-review
**Slash Commands**:
- Universal: /file-write, /memory-store
- Specialist: /architecture-design, /system-diagram, /tech-stack-selection, /scalability-plan

**MCP Tools**:
- mcp__claude-flow__memory_store (store architecture)
- mcp__ruv-swarm__swarm_init (architecture topology)
- mcp__flow-nexus__workflow_create (architecture workflow)

**Triggers**: When need system design, architecture decisions, tech stack selection

---

### 11. DevOps / CI-CD Engineer
**Used In**: sop-product-launch, sop-api-development, sop-code-review
**Slash Commands**:
- Universal: /git-push, /communicate-notify, /test-run
- Specialist: /pipeline-setup, /deployment, /docker-build, /k8s-deploy, /terraform-plan

**MCP Tools**:
- mcp__flow-nexus__sandbox_create (deployment simulation)
- mcp__flow-nexus__workflow_create (CI/CD workflow)
- mcp__flow-nexus__workflow_execute (run deployment)
- mcp__claude-flow__memory_retrieve (deployment config)

**Triggers**: When need CI/CD pipeline, Docker deployment, Kubernetes orchestration

---

### 12. Performance Analyzer
**Used In**: sop-product-launch, sop-api-development, sop-code-review
**Slash Commands**:
- Universal: /test-run, /communicate-report
- Specialist: /load-test, /stress-test, /benchmark, /profiling

**MCP Tools**:
- mcp__ruv-swarm__benchmark_run (performance testing)
- mcp__ruv-swarm__agent_metrics (agent performance)
- mcp__flow-nexus__neural_performance_benchmark (neural benchmarks)
- mcp__claude-flow__memory_store (performance metrics)

**Triggers**: When need performance optimization, load testing, bottleneck identification

---

### 13. Security Manager
**Used In**: sop-product-launch, sop-api-development, sop-code-review
**Slash Commands**:
- Universal: /file-read, /communicate-report
- Specialist: /security-audit, /owasp-check, /dependency-audit, /secrets-scan

**MCP Tools**:
- mcp__flow-nexus__sandbox_execute (security testing)
- mcp__ruv-swarm__agent_spawn (security sub-agents)
- mcp__claude-flow__memory_store (security findings)

**Triggers**: When need security review, OWASP compliance, vulnerability management

---

### 14. API Documentation Specialist
**Used In**: sop-product-launch, sop-api-development, sop-code-review
**Slash Commands**:
- Universal: /file-read, /file-write, /markdown-gen
- Specialist: /openapi-generate, /api-reference, /code-examples, /tutorial-create

**MCP Tools**:
- mcp__claude-flow__memory_retrieve (API specs)
- mcp__flow-nexus__template_deploy (docs template)
- mcp__ruv-swarm__agent_spawn (docs sub-agents)

**Triggers**: When need API documentation, developer guides, technical writing

---

### 15. Production Validator
**Used In**: sop-product-launch, sop-api-development
**Slash Commands**:
- Universal: /test-run, /test-validate, /communicate-report
- Specialist: /production-checklist, /go-no-go-decision, /rollback-plan

**MCP Tools**:
- mcp__flow-nexus__sandbox_execute (validation tests)
- mcp__ruv-swarm__benchmark_run (final benchmarks)
- mcp__claude-flow__memory_retrieve (all validation data)

**Triggers**: When need production readiness check, deployment validation

---

### 16. Performance Monitor
**Used In**: sop-product-launch
**Slash Commands**:
- Universal: /communicate-report, /communicate-log
- Specialist: /metrics-collect, /alert-configure, /dashboard-create, /anomaly-detect

**MCP Tools**:
- mcp__ruv-swarm__swarm_monitor (real-time monitoring)
- mcp__ruv-swarm__agent_metrics (agent health)
- mcp__flow-nexus__system_health (system status)
- mcp__claude-flow__memory_store (metrics history)

**Triggers**: When need production monitoring, real-time alerts, metrics dashboards

---

### 17. Code Reviewer / Code Analyzer
**Used In**: sop-api-development, sop-code-review
**Slash Commands**:
- Universal: /file-read, /communicate-report
- Specialist: /code-quality-check, /style-audit, /complexity-analyze, /best-practices-check

**MCP Tools**:
- mcp__flow-nexus__sandbox_execute (run static analysis)
- mcp__claude-flow__memory_store (review feedback)
- mcp__ruv-swarm__agent_spawn (specialized reviewers)

**Triggers**: When need code review, quality assessment, best practices validation

---

### 18. PR Manager / GitHub Modes
**Used In**: sop-code-review
**Slash Commands**:
- Universal: /git-status, /git-diff, /communicate-notify
- Specialist: /pr-create, /pr-review, /pr-merge, /pr-label

**MCP Tools**:
- mcp__flow-nexus__github_repo_analyze (PR analysis)
- mcp__ruv-swarm__task_orchestrate (coordinate reviewers)
- mcp__claude-flow__memory_store (review state)

**Triggers**: When need PR management, code review coordination, merge decisions

---

### 19. Marketing Specialist
**Used In**: sop-product-launch
**Slash Commands**:
- Universal: /file-write, /memory-store, /agent-delegate
- Specialist: /campaign-create, /audience-segment, /ab-test-setup, /funnel-analyze

**MCP Tools**:
- mcp__ruv-swarm__agent_spawn (campaign sub-agents)
- mcp__flow-nexus__workflow_create (campaign workflow)
- mcp__claude-flow__memory_store (campaign data)
- mcp__ruv-swarm__neural_train (learn from campaigns)

**Triggers**: When need marketing campaigns, audience analysis, conversion optimization

---

### 20. Sales Specialist
**Used In**: sop-product-launch
**Slash Commands**:
- Universal: /file-write, /memory-store, /communicate-notify
- Specialist: /pipeline-manage, /lead-qualify, /forecast-generate, /proposal-create

**MCP Tools**:
- mcp__ruv-swarm__agent_spawn (sales sub-agents)
- mcp__claude-flow__memory_store (sales data)
- mcp__flow-nexus__workflow_create (sales workflow)

**Triggers**: When need sales pipeline, lead management, revenue forecasting

---

### 21. Customer Support Specialist
**Used In**: sop-product-launch
**Slash Commands**:
- Universal: /file-read, /communicate-notify, /memory-retrieve
- Specialist: /ticket-triage, /knowledge-base-create, /support-response, /escalation-manage

**MCP Tools**:
- mcp__ruv-swarm__agent_spawn (support sub-agents)
- mcp__claude-flow__memory_retrieve (knowledge base)
- mcp__flow-nexus__realtime_subscribe (live ticket updates)

**Triggers**: When need customer support, ticket management, help desk operations

---

### 22. Content Creator
**Used In**: sop-product-launch
**Slash Commands**:
- Universal: /file-write, /markdown-gen
- Specialist: /blog-create, /social-post, /email-sequence, /video-script

**MCP Tools**:
- mcp__claude-flow__memory_retrieve (content strategy)
- mcp__flow-nexus__template_deploy (content templates)
- mcp__ruv-swarm__neural_train (learn content patterns)

**Triggers**: When need blog posts, social media content, email campaigns

---

### 23. SEO Specialist
**Used In**: sop-product-launch
**Slash Commands**:
- Universal: /file-read, /file-edit, /communicate-report
- Specialist: /keyword-research, /on-page-seo, /link-building, /seo-audit

**MCP Tools**:
- mcp__flow-nexus__github_repo_analyze (analyze site structure)
- mcp__claude-flow__memory_store (SEO data)

**Triggers**: When need search optimization, keyword strategy, organic traffic growth

---

### 24. System Integrator
**Used In**: sop-product-launch
**Slash Commands**:
- Universal: /test-run, /communicate-report
- Specialist: /integration-test, /api-integration, /third-party-setup

**MCP Tools**:
- mcp__flow-nexus__sandbox_execute (integration testing)
- mcp__ruv-swarm__task_orchestrate (coordinate integrations)
- mcp__claude-flow__memory_retrieve (all component specs)

**Triggers**: When need component integration, API connections, third-party services

---

## MECE Validation

### Mutually Exclusive (No Overlap)
✅ Each agent has distinct responsibilities
✅ No command duplication across specialist agents
✅ Clear boundaries between agent domains

### Collectively Exhaustive (Complete Coverage)
✅ All SOP workflow roles covered
✅ Development lifecycle complete (plan → build → test → deploy → monitor)
✅ Business functions complete (marketing, sales, support)
✅ Technical functions complete (backend, frontend, mobile, database, DevOps)

---

## Naming Conventions

### Agent Names (Simple Role/Function)
**Pattern**: `{role}-{specialization}` (e.g., "marketing-specialist", "backend-developer")

**Examples**:
- ✅ `marketing-specialist`
- ✅ `sales-specialist`
- ✅ `backend-developer`
- ✅ `security-manager`
- ✅ `performance-analyzer`

### Skill Names (Trigger-Based)
**Pattern**: `{action}-{domain}-when-{trigger}`

**Examples**:
- ❌ Old: `agent-creator`
- ✅ New: `create-specialized-agent-when-need-new-domain-expert`

- ❌ Old: `sop-product-launch`
- ✅ New: `orchestrate-product-launch-when-releasing-new-product`

- ❌ Old: `sop-api-development`
- ✅ New: `build-rest-api-when-need-backend-service`

- ❌ Old: `sop-code-review`
- ✅ New: `review-code-comprehensively-when-pr-submitted`

**Key Distinction**:
- **Agents** = WHO does the work (simple names)
- **Skills** = WHEN to use them (trigger-based names)

---

## Agent Priority for Rewrites

### Tier 1: Business Critical (Immediate)
1. **Marketing Specialist** - Highest ROI potential
2. **Sales Specialist** - Revenue generation
3. **Backend Developer** - Most frequently used
4. **DevOps Engineer** - Deployment critical
5. **Security Manager** - Risk mitigation

### Tier 2: Technical Foundation (Week 2)
6. Frontend Developer
7. Mobile Developer
8. Database Architect
9. QA Engineer
10. System Architect

### Tier 3: Specialized Support (Week 3)
11. API Documentation Specialist
12. Customer Support Specialist
13. Performance Analyzer
14. Code Reviewer
15. Product Manager

### Tier 4: Workflow Specific (Week 4)
16. Market Researcher
17. Business Analyst
18. Content Creator
19. SEO Specialist
20. Performance Monitor
21. Production Validator
22. PR Manager
23. System Integrator
24. Support Specialist

---

## Command Assignment Matrix

### Universal Commands (ALL agents get these)
- File: read, write, edit, glob-search, grep-search
- Git: status, diff, commit, push, branch
- Communication: notify, report, log, delegate, escalate
- Memory: store, retrieve, search
- Testing: test-run, coverage, validate

### Specialist Command Distribution
- **Development**: api-design, component-create, db-migrate, schema-design
- **Marketing**: campaign-create, audience-segment, ab-test-setup, funnel-analyze
- **Sales**: pipeline-manage, lead-qualify, forecast-generate
- **DevOps**: pipeline-setup, deployment, docker-build, k8s-deploy
- **Security**: security-audit, vulnerability-scan, penetration-test
- **Testing**: test-suite-create, integration-test, e2e-test
- **Documentation**: openapi-generate, api-reference, tutorial-create

### MCP Tool Distribution
- **Coordination** (ALL agents): swarm_init, agent_spawn, task_orchestrate
- **Memory** (ALL agents): memory_store, memory_retrieve, memory_search
- **Sandbox** (Dev agents): sandbox_create, sandbox_execute, sandbox_configure
- **Workflow** (Orchestrator agents): workflow_create, workflow_execute
- **Monitoring** (Ops agents): swarm_monitor, agent_metrics, system_health
- **Neural** (Learning agents): neural_train, neural_patterns, daa_agent_create

---

**Status**: MECE inventory complete
**Total Unique Agents**: 24
**Ready for**: Enhanced agent-creator rewrites
**Next**: Begin with Tier 1 agents using new naming convention
