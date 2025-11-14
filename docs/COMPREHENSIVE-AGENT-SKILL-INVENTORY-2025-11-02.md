# Comprehensive Agent & Skill Inventory
**Date**: 2025-11-02
**System**: ruv-sparc-three-loop-system v2.0.0
**Status**: Agent Discovery Complete | Skills Cataloged

---

## 🎯 CRITICAL DISTINCTION: Agents vs Skills

### AGENTS = WHO (Autonomous Actors)
- **Definition**: Specialized AI workers with specific capabilities
- **Location**: `claude-code-plugins/ruv-sparc-three-loop-system/agents/`
- **Format**: Individual .md files with agent definitions
- **Purpose**: Execute tasks autonomously with defined SOPs

### SKILLS = WHAT (Workflows/Processes)
- **Definition**: Reusable workflows that coordinate agents
- **Location**: `.claude/skills/` and `plugin/skills/`
- **Format**: Directories with SKILL.md and supporting files
- **Purpose**: Orchestrate multi-agent workflows for complex tasks

---

## 📊 THE TRUE COUNT

### Agent Inventory

| Metric | Count | Status |
|--------|-------|--------|
| **Total Agents (Current)** | **130** | ✅ 65% of target |
| **Target Agent Count** | **200** | 🎯 Goal |
| **Agent .md Files** | 130 | Physical files |
| **Agent Categories** | 25 | Organized domains |
| **Batch 1-4 Complete** | 130 agents | ✅ Done |
| **Batch 5 Planned** | 35 agents (131-165) | 📋 Nov 15 |
| **Batch 6 Planned** | 35 agents (166-200) | 📋 Dec 1 |

### Skill Inventory

| Metric | Count | Status |
|--------|-------|--------|
| **Total Skills (Unique)** | **~113** | ✅ Complete |
| **Skills in .claude/skills/** | 112 | Active workspace (+3 research) |
| **Skills in plugin repo** | 112 | Plugin system |
| **Core Skills (SPARC)** | 13 | Methodology |
| **Three-Loop Skills** | 3 | Loop 1-3 |
| **Testing Skills** | 9 | QA workflows |
| **GitHub Skills** | 14 | Repository management |

---

## 🗂️ Agent Registry (130 Agents Across 17 Domains)

### Domain 1: Core Development (8 agents)
**Location**: `agents/core/`
1. **coder** - Implementation specialist (code generation, refactoring)
2. **coder-enhanced** - Advanced coder with optimization
3. **reviewer** - Code review specialist (quality assurance)
4. **tester** - Testing specialist (≥90% coverage, TDD)
5. **planner** - Task planning and coordination
6. **researcher** - Research and information gathering
7. **api-designer** - REST/GraphQL API design
8. **technical-debt-manager** - Tech debt identification and refactoring

**Commands**: 92 total | Avg 11.5 per agent
**Coverage**: Core development workflow (design→code→test→review)

---

### Domain 2: Swarm Coordination (15 agents)
**Location**: `agents/swarm/`, `agents/consensus/`
1. **hierarchical-coordinator** - Tree topology coordination
2. **mesh-coordinator** - Peer-to-peer coordination
3. **adaptive-coordinator** - Dynamic topology adjustment
4. **ring-coordinator** - Circular topology
5. **collective-intelligence-coordinator** - Hive mind orchestration
6. **swarm-memory-manager** - Shared memory management
7. **consensus-validator** - Byzantine agreement, quorum validation
8. **swarm-health-monitor** - Health monitoring, failure detection
9. **byzantine-coordinator** - Byzantine fault tolerance
10. **raft-manager** - Raft consensus protocol
11. **gossip-coordinator** - Gossip protocol coordination
12. **crdt-synchronizer** - Conflict-free replicated data
13. **quorum-manager** - Quorum consensus
14. **security-manager** - Security coordination
15. **consensus-builder** - General consensus mechanisms

**Commands**: 173 total | Avg 11.5 per agent
**Coverage**: Multi-agent coordination, distributed systems

---

### Domain 3: Testing & Validation (9 agents) 🆕
**Location**: `agents/testing/`
1. **tdd-london-swarm** - London-school TDD
2. **production-validator** - Production readiness validation
3. **e2e-testing-specialist** - Playwright, Cypress, end-to-end automation
4. **performance-testing-agent** - k6, JMeter, load/stress testing
5. **security-testing-agent** - SAST, DAST, vulnerability scanning
6. **visual-regression-agent** - Screenshot comparison, UI testing
7. **contract-testing-agent** - Pact, API contract validation
8. **chaos-engineering-agent** - Fault injection, resilience testing
9. **audit-pipeline-orchestrator** - Complete audit workflow

**Commands**: 108 total | Avg 12.0 per agent
**Coverage**: Comprehensive testing (unit→integration→e2e→security→chaos)

---

### Domain 4: Frontend Development (6 agents) 🆕
**Location**: `agents/frontend/`
1. **react-developer** - React 18, hooks, state management
2. **vue-developer** - Vue.js, Composition API, Pinia
3. **ui-component-builder** - Design systems, component libraries
4. **css-styling-specialist** - Tailwind, styled-components, CSS-in-JS
5. **frontend-accessibility-auditor** - WCAG, screen readers, a11y testing
6. **frontend-performance-optimizer** - Lighthouse, Core Web Vitals, bundles

**Commands**: 75 total | Avg 12.5 per agent
**Coverage**: Modern frontend development (React/Vue + performance + a11y)

---

### Domain 5: Database & Data Layer (7 agents) 🆕
**Location**: `agents/database/`, `agents/data/`
1. **database-design-specialist** - Schema design, normalization, ER diagrams
2. **query-optimization-agent** - SQL tuning, index optimization
3. **database-migration-agent** - Schema migrations, zero-downtime
4. **data-pipeline-engineer** - ETL, Airflow, Kafka, stream processing
5. **cache-strategy-agent** - Redis, Memcached, caching patterns
6. **database-backup-recovery-agent** - Backup strategies, disaster recovery
7. **data-ml-model** - ML data pipelines

**Commands**: 91 total | Avg 13.0 per agent
**Coverage**: Complete data layer (design→optimize→migrate→pipeline→cache→backup)

---

### Domain 6: Documentation & Knowledge (6 agents) 🆕
**Location**: `agents/documentation/`
1. **documentation-specialist** - README, setup guides, architecture docs
2. **api-documentation-specialist** - OpenAPI, AsyncAPI, interactive docs
3. **knowledge-base-manager** - Documentation organization, search, versioning
4. **technical-writer-agent** - Blog posts, tutorials, case studies
5. **diagram-generator-agent** - System diagrams, C4 models, UML
6. **docs-api-openapi** - OpenAPI/Swagger specialist

**Commands**: 70 total | Avg 11.7 per agent
**Coverage**: Complete documentation lifecycle

---

### Domain 7: GitHub & Repository (14 agents)
**Location**: `agents/github/`
1. **github-modes** - GitHub operation modes
2. **pr-manager** - Pull request management
3. **code-review-swarm** - Multi-agent PR review
4. **issue-tracker** - Issue management
5. **release-manager** - Release automation
6. **workflow-automation** - GitHub Actions automation
7. **project-board-sync** - Project board synchronization
8. **repo-architect** - Repository architecture
9. **multi-repo-swarm** - Multi-repository coordination
10. **cicd-engineer** - CI/CD pipeline specialist
11. **cicd-github-integration** - GitHub CI/CD integration
12. **release-automation** - Automated release workflows
13. **pr-review-automation** - Automated PR review
14. **issue-automation** - Automated issue triaging

**Commands**: 161 total | Avg 11.5 per agent
**Coverage**: Complete GitHub workflow automation

---

### Domain 8: Business & Product (8 agents)
**Location**: `agents/business/`
1. **business-analyst** - SWOT, business model, revenue projection
2. **product-manager** - Roadmap, feature prioritization, user stories
3. **market-researcher** - Competitive analysis, customer insights
4. **content-creator** - Content creation and management
5. **customer-support-specialist** - Customer support workflows
6. **marketing-specialist** - Marketing campaigns and strategies
7. **sales-specialist** - Sales processes and automation
8. **seo-specialist** - SEO optimization and analysis

**Commands**: 88 total | Avg 11.0 per agent
**Coverage**: Business operations (product→marketing→sales→support)

---

### Domain 9: SPARC Methodology (13 agents)
**Location**: `agents/sparc/`
1. **sparc-coord** - SPARC workflow coordination
2. **sparc-coder** - SPARC-based coding
3. **specification** - Requirements specification phase
4. **pseudocode** - Algorithm design phase
5. **architecture** - System architecture phase
6. **refinement** - TDD refinement phase
7. **sparc-tester** - SPARC testing phase
8. **sparc-reviewer** - SPARC review phase
9. **sparc-planner** - SPARC planning phase
10. **sparc-researcher** - SPARC research phase
11. **sparc-analyzer** - SPARC analysis phase
12. **sparc-integrator** - SPARC integration phase
13. **sparc-validator** - SPARC validation phase

**Commands**: 150 total | Avg 11.5 per agent
**Coverage**: Complete SPARC methodology (Spec→Pseudocode→Arch→Refine→Complete)

---

### Domain 10: Specialized Development (11 agents)
**Location**: `agents/development/`
1. **backend-dev** - Backend API development
2. **dev-backend-api-enhanced** - Advanced backend development
3. **mobile-dev** - React Native, iOS, Android
4. **ml-developer** - ML pipeline development
5. **frontend-dev** - Frontend development specialist
6. **fullstack-dev** - Full-stack development
7. **devops-specialist** - DevOps and infrastructure
8. **cloud-architect** - Cloud architecture
9. **microservices-architect** - Microservices design
10. **api-gateway-specialist** - API gateway configuration
11. **container-orchestration** - Kubernetes, Docker Swarm

**Commands**: 127 total | Avg 11.5 per agent
**Coverage**: Specialized development disciplines

---

### Domain 11: Research & Analysis (6 agents)
**Location**: `agents/research/`, `agents/analysis/`
1. **data-steward** - Dataset documentation, bias auditing, DVC
2. **ethics-agent** - Ethics review, risk assessment, safety evaluation
3. **archivist** - Reproducibility, DOI, model card, artifact archival
4. **evaluator** - Quality gate validation, GO/NO-GO decisions
5. **root-cause-analyzer** - Root cause analysis, debugging
6. **code-analyzer** - Code quality analysis

**Commands**: 69 total | Avg 11.5 per agent
**Coverage**: Research lifecycle + quality gates

---

### Domain 12: Template & Meta (9 agents)
**Location**: `agents/templates/`, `agents/meta/`
1. **base-template-generator** - Clean foundational templates
2. **template-orchestrator** - Template workflow coordination
3. **skill-creator-agent** - Skill creation specialist
4. **agent-creator** - Agent creation with 4-phase SOP
5. **prompt-architect** - Prompt engineering and optimization
6. **workflow-designer** - Workflow design and optimization
7. **meta-coordinator** - Meta-level coordination
8. **skill-enhancer** - Skill enhancement and improvement
9. **agent-enhancer** - Agent enhancement and optimization

**Commands**: 104 total | Avg 11.6 per agent
**Coverage**: Meta-programming (creating agents, skills, templates)

---

### Domain 13: Flow-Nexus Platform (7 agents)
**Location**: `agents/flow-nexus/`
1. **authentication** - Flow-Nexus authentication
2. **sandbox-manager** - Sandbox lifecycle management
3. **neural-trainer** - Neural network training
4. **workflow-orchestrator** - Workflow execution
5. **storage-manager** - Cloud storage management
6. **realtime-coordinator** - Real-time event coordination
7. **seraphina-interface** - Queen Seraphina AI assistant

**Commands**: 81 total | Avg 11.6 per agent
**Coverage**: Cloud platform features (authentication→sandboxes→neural→storage)

---

### Domain 14: Performance & Optimization (5 agents)
**Location**: `agents/optimization/`, `agents/consensus/`
1. **perf-analyzer** - Performance analysis and bottleneck detection
2. **performance-benchmarker** - Performance benchmarking
3. **performance-benchmarker-enhanced** - Advanced benchmarking
4. **load-balancer** - Load balancing strategies
5. **memory-coordinator** - Memory optimization

**Commands**: 58 total | Avg 11.6 per agent
**Coverage**: Performance optimization

---

### Domain 15: Root-Level Utility (5 agents)
**Location**: `agents/utility/`
1. **smart-agent** - Multi-purpose intelligent agent
2. **migration-planner** - Migration planning and execution
3. **swarm-init** - Swarm initialization
4. **task-orchestrator** - Task orchestration
5. **utility-coordinator** - General utility coordination

**Commands**: 58 total | Avg 11.6 per agent
**Coverage**: General utilities

---

### Domain 16: Audit & Validation (4 agents)
**Location**: `agents/analysis/`
1. **audit-pipeline-orchestrator** - Complete audit workflow
2. **functionality-audit** - Sandbox testing + systematic debugging
3. **theater-detection-audit** - 6-agent Byzantine consensus verification
4. **production-readiness-validator** - Deployment readiness validation

**Commands**: 46 total | Avg 11.5 per agent
**Coverage**: Quality audit pipeline

---

### Domain 17: Infrastructure & Cloud (0 agents) ⏳
**Status**: PLANNED for Batch 5 (Agents 131-165)
**Target**: 10 agents
**Planned Agents**:
- kubernetes-specialist
- terraform-iac-specialist
- aws-specialist
- gcp-specialist
- azure-specialist
- docker-specialist
- terraform-specialist
- ansible-specialist
- cloud-security-specialist
- infrastructure-monitoring

---

## 📁 Skill Registry (~110 Skills)

### Three-Loop System Skills (3 skills)
1. **research-driven-planning** (Loop 1) - Research + 5x pre-mortem + risk mitigation
2. **parallel-swarm-implementation** (Loop 2) - 9-step parallel swarm development
3. **cicd-intelligent-recovery** (Loop 3) - Automated failure recovery + root cause

**Uses**: hierarchical-coordinator, researcher, coder, tester, reviewer, planner

---

### Core Development Skills (13 skills)
1. **agent-creator** - 4-phase SOP agent creation
2. **sparc-methodology** - 5-phase SPARC workflow
3. **functionality-audit** - Sandbox testing + debugging
4. **theater-detection-audit** - 6-agent Byzantine verification
5. **production-readiness** - Deployment checklist
6. **code-review-assistant** - Multi-agent PR review
7. **quick-quality-check** - Parallel lint/security/tests
8. **smart-bug-fix** - Intelligent debugging
9. **prompt-architect** - Prompt engineering
10. **skill-builder** - Skill creation
11. **micro-skill-creator** - Atomic skill creation
12. **slash-command-encoder** - Slash command creation
13. **base-template-generator** - Template generation

**Uses**: coder, reviewer, tester, planner, researcher, agent-creator

---

### Testing & Quality Skills (9 skills)
1. **testing-quality** - TDD framework
2. **tdd-london-swarm** - London-school TDD
3. **e2e-testing** - End-to-end automation
4. **performance-testing** - Load/stress testing
5. **security-testing** - SAST/DAST scanning
6. **visual-regression-testing** - UI testing
7. **contract-testing** - API contract validation
8. **chaos-engineering** - Fault injection testing
9. **ml-training-debugger** - ML-specific debugging

**Uses**: tester, e2e-testing-specialist, performance-testing-agent, security-testing-agent, visual-regression-agent, contract-testing-agent, chaos-engineering-agent

---

### Swarm & Coordination Skills (14 skills)
1. **swarm-orchestration** - Swarm topology + coordination
2. **swarm-advanced** - Advanced swarm features
3. **hive-mind-advanced** - Queen-led coordination
4. **coordination** - Agent coordination patterns
5. **flow-nexus-platform** - Cloud orchestration
6. **flow-nexus-swarm** - Cloud swarm deployment
7. **flow-nexus-neural** - Cloud neural training
8. **ruv-swarm-coordination** - RUV swarm features
9. **hierarchical-coordination** - Tree topology
10. **mesh-coordination** - Peer-to-peer
11. **adaptive-coordination** - Dynamic topology
12. **ring-coordination** - Circular topology
13. **byzantine-consensus** - Byzantine fault tolerance
14. **raft-consensus** - Raft protocol

**Uses**: All 15 swarm coordination agents

---

### GitHub Skills (14 skills)
1. **github-code-review** - AI swarm PR review
2. **github-project-management** - Issue tracking + boards
3. **github-workflow-automation** - CI/CD pipelines
4. **github-release-management** - Versioning + deployment
5. **github-multi-repo** - Multi-repo coordination
6. **github-integration** - General GitHub operations
7. **sop-product-launch** - Complete launch workflow
8. **sop-code-review** - Systematic code review
9. **pr-automation** - Automated PR workflows
10. **issue-automation** - Automated issue triaging
11. **release-automation** - Automated releases
12. **workflow-cicd** - CI/CD workflow automation
13. **repo-sync** - Repository synchronization
14. **branch-management** - Branch strategy automation

**Uses**: All 14 GitHub agents

---

### Specialized Development Skills (20+ skills)
**Backend/API**:
- when-building-backend-api-orchestrate-api-development
- sop-api-development

**Frontend**:
- react-specialist
- vue-specialist
- ui-component-builder

**Database**:
- sql-database-specialist
- database-design
- query-optimization

**Mobile**:
- mobile-dev

**ML/AI**:
- ml-expert
- ml-developer
- agentdb
- agentdb-memory-patterns
- agentdb-learning
- agentdb-optimization
- agentdb-vector-search
- agentdb-advanced

**Infrastructure**:
- docker-containerization
- kubernetes-specialist
- terraform-iac
- aws-specialist
- opentelemetry-observability

**Uses**: backend-dev, frontend-dev, mobile-dev, ml-developer, database agents

---

### Documentation Skills (5 skills)
1. **pptx-generation** - Enterprise PPT generation
2. **documentation** - Documentation generation
3. **api-documentation** - API docs (OpenAPI/AsyncAPI)
4. **technical-writing** - Blog posts, tutorials
5. **diagram-generation** - System diagrams, C4, UML

**Uses**: documentation-specialist, api-documentation-specialist, technical-writer-agent, diagram-generator-agent

---

### Security & Compliance Skills (5 skills)
1. **network-security-setup** - Sandbox isolation
2. **security** - Security scanning + fixes
3. **wcag-accessibility** - WCAG 2.1 AA/AAA compliance
4. **reverse-engineering-quick** - Malware triage
5. **reverse-engineering-deep** - Advanced malware analysis

**Uses**: security-testing-agent, security-manager, frontend-accessibility-auditor

---

### Foundational Research Skills (3 skills) 🆕 PRIORITY 1
1. **general-research-workflow** - Red's 6-phase research methodology with 7 Quality Gates
2. **academic-reading-workflow** - Blue's searchable annotation system ("Command-F in Real Life")
3. **source-credibility-analyzer** - Automated source scoring with program-of-thought rubrics

**Uses**: researcher, analyst, coordinator
**Duration**: 2-40 hours (general-research), 2-6 hours (academic-reading), 5-15 min (source-credibility)
**Key Innovation**: Program-of-thought rubrics for transparent, auditable scoring

---

### Deep Research SOP Skills (9 skills)
1. **baseline-replication** - Baseline replication with ±1% tolerance
2. **literature-synthesis** - PRISMA 2020 systematic review
3. **method-development** - Novel algorithm development
4. **holistic-evaluation** - Multi-metric evaluation
5. **deployment-readiness** - Production ML deployment
6. **deep-research-orchestrator** - Complete research pipeline
7. **reproducibility-audit** - ACM artifact evaluation
8. **research-publication** - Paper writing + submission
9. **gate-validation** - Quality gate validation

**Uses**: data-steward, ethics-agent, archivist, evaluator

---

### Self-Improvement Skills (Dogfooding) (3 skills)
1. **sop-dogfooding-quality-detection** - Connascence analysis
2. **sop-dogfooding-pattern-retrieval** - Pattern search
3. **sop-dogfooding-continuous-improvement** - Full improvement cycle

**Uses**: code-analyzer, reviewer, coder, hierarchical-coordinator

---

### Intelligence & Learning Skills (5 skills)
1. **reasoningbank-intelligence** - Pattern recognition
2. **reasoningbank-agentdb** - Trajectory tracking
3. **agentdb** - Vector search
4. **agentdb-learning** - Reinforcement learning
5. **agentdb-optimization** - Memory optimization

**Uses**: All agents with learning capabilities

---

### Automation & Workflow Skills (8 skills)
1. **hooks-automation** - Hook integration
2. **workflow** - Workflow creation
3. **i18n-automation** - Internationalization
4. **stream-chain** - Streaming workflows
5. **cascade-orchestrator** - Sequential/parallel/conditional chains
6. **feature-dev-complete** - 12-stage end-to-end feature development
7. **pair-programming** - Driver/Navigator/Switch modes
8. **intent-analyzer** - Intent analysis + Socratic clarification

**Uses**: Various agents based on task type

---

### Analysis & Debugging Skills (6 skills)
1. **performance-analysis** - Performance analysis
2. **style-audit** - Code style analysis
3. **verification-quality** - Quality verification
4. **dependencies** - Dependency analysis
5. **debugging** - Systematic debugging
6. **reverse-engineer-debug** - Code comprehension + debugging

**Uses**: perf-analyzer, code-analyzer, reviewer

---

## 🔄 Skill-Agent Mapping

### Pattern: Skills COORDINATE Agents

**Example: research-driven-planning (Loop 1)**
```
Skill: research-driven-planning
├── Phase 1: Specification
│   └── Uses: specification agent
├── Phase 2: Research
│   └── Uses: researcher agent
├── Phase 3: Planning
│   └── Uses: planner agent
├── Phase 4: Pre-mortem (5 iterations)
│   └── Uses: reviewer agent, researcher agent
└── Phase 5: Knowledge Storage
    └── Uses: archivist agent
```

**Example: parallel-swarm-implementation (Loop 2)**
```
Skill: parallel-swarm-implementation
├── Step 1: Init
│   └── Uses: hierarchical-coordinator
├── Step 2: Discovery
│   └── Uses: code-analyzer, researcher
├── Step 3: MECE
│   └── Uses: planner
├── Step 4: Deploy
│   └── Uses: coder, backend-dev, frontend-dev, mobile-dev (parallel)
├── Step 5: Theater Detection
│   └── Uses: theater-detection-audit skill → 6 reviewer agents
├── Step 6: Integrate
│   └── Uses: coder, tester
├── Step 7: Audit
│   └── Uses: audit-pipeline-orchestrator
├── Step 8: Knowledge
│   └── Uses: archivist
└── Step 9: Complete
    └── Uses: production-validator
```

**Example: github-code-review**
```
Skill: github-code-review
└── Uses: code-review-swarm
    ├── security-reviewer agent
    ├── performance-reviewer agent
    ├── style-reviewer agent
    ├── test-reviewer agent
    └── documentation-reviewer agent
```

---

## 📈 Growth Roadmap

### Current Status (2025-11-02)
- ✅ 130 agents (65% of 200-agent target)
- ✅ ~110 skills (complete)
- ✅ 100% command coverage (58/58 commands)
- ✅ 17 domains mapped
- ✅ 70% technology stack coverage

### Batch 5 (Target: 2025-11-15) - 35 agents
**Focus**: Infrastructure, Audits, AI/ML, Business
- 10 Infrastructure & Cloud agents
- 6 Audit & Compliance agents
- 5 AI/ML Specialization agents
- 4 Business & Product agents
- 4 Research & Analysis agents
- 3 Template & Meta agents
- 2 GitHub Enhancement agents
- 1 Incident Response agent

### Batch 6 (Target: 2025-12-01) - 35 agents
**Focus**: Specialization, Optimization, Advanced
- 14 Specialized Development agents
- 5 Testing Enhancement agents
- 3 Security Enhancement agents
- 3 Optimization agents
- 10 Advanced Specialists

### Final Target
- 🎯 200 total agents
- 🎯 ~120 total skills
- 🎯 100% command coverage maintained
- 🎯 100% technology stack coverage

---

## 🎯 Key Insights

### 1. Agents vs Skills Separation is CRITICAL
- **Agents** = Autonomous workers (130 currently)
- **Skills** = Workflows that coordinate agents (~110 currently)
- **Relationship**: Skills USE agents to accomplish complex tasks

### 2. Three-Loop System Architecture
- **Loop 1** (research-driven-planning): Uses 5+ agents for risk-mitigated planning
- **Loop 2** (parallel-swarm-implementation): Uses 11+ agents for theater-free development
- **Loop 3** (cicd-intelligent-recovery): Uses 4+ agents for intelligent recovery

### 3. Agent Specialization Levels
- **Core agents** (8): General-purpose development (coder, reviewer, tester)
- **Specialist agents** (122): Domain-specific expertise (react-developer, database-design-specialist)
- **Coordination agents** (15): Multi-agent orchestration (hierarchical-coordinator, consensus-validator)

### 4. Skill Complexity Levels
- **Simple skills** (~40): Single-agent workflows (smart-bug-fix, documentation)
- **Medium skills** (~50): 2-5 agent workflows (code-review-assistant, testing-quality)
- **Complex skills** (~20): 5+ agent workflows (three-loop system, deep-research-orchestrator)

---

## 📚 Documentation References

**Agent Registry**: `docs/AGENT-REGISTRY.md`
**Skill Inventory**: `docs/COMPLETE-SKILLS-INVENTORY.md`
**MECE Agent Taxonomy**: `docs/agent-taxonomy/MECE-AGENT-TAXONOMY.md`
**MECE Skills Taxonomy**: `docs/skills-taxonomy/MECE-SKILLS-TAXONOMY.md`
**Three-Loop Integration**: `.claude/skills/THREE-LOOP-INTEGRATION-ARCHITECTURE.md`
**Agent-Skill Mappings**: `docs/SKILL-AGENT-COMMAND-MAPPINGS.md`

---

**Report Version**: 1.0.0
**Last Updated**: 2025-11-02
**Maintained By**: SPARC System + Intent Analyzer
**Next Review**: 2025-11-15 (Batch 5 completion)
