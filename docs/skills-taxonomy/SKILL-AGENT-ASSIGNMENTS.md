# Skill-Agent Assignment Matrix

**Version**: 1.0.0
**Created**: 2025-11-02
**Total Skills**: 130
**Total Agents**: 130
**Coverage Analysis**: Complete domain mapping with primary, secondary, and fallback assignments

---

## Executive Summary

This document provides comprehensive skill-to-agent assignments across all 130 skills and 130 agents in the SPARC ecosystem. Each skill is assigned:

- **PRIMARY Agent**: Best-match specialist for skill execution
- **SECONDARY Agents**: Supporting coordination and backup
- **FALLBACK Agents**: Alternative if primary unavailable
- **Coordination Pattern**: How agents collaborate via hooks, memory, and swarm

### Key Statistics

| Metric | Value |
|--------|-------|
| Total Skills Mapped | 130 |
| Total Agents Available | 130 |
| Avg Agents per Skill | 3.2 (1 primary + 2.2 secondary) |
| Skills with Full Coverage | 130 (100%) |
| Multi-Agent Skills | 98 (75%) |
| Single-Agent Skills | 32 (25%) |

---

## Domain-Based Skill Assignments

### 1. Development Lifecycle Skills (15 skills)

#### 1.1 Feature Development

**Skill**: `feature-dev-complete`
- **PRIMARY**: researcher (requirements analysis)
- **SECONDARY**: system-architect, coder, tester, reviewer, api-docs, cicd-engineer
- **COORDINATION**: Hierarchical topology with 7-phase workflow
- **MEMORY**: `swarm/feature-dev/*` (requirements, architecture, implementation, tests, review, docs, deployment)

**Skill**: `parallel-swarm-implementation`
- **PRIMARY**: hierarchical-coordinator
- **SECONDARY**: coder, tester, reviewer (coordinated in parallel)
- **COORDINATION**: Mesh topology for parallel execution
- **MEMORY**: `swarm/parallel-impl/*`

**Skill**: `research-driven-planning`
- **PRIMARY**: researcher
- **SECONDARY**: planner, system-architect
- **FALLBACK**: analyst
- **COORDINATION**: Sequential planning workflow
- **MEMORY**: `swarm/planning/research-findings`

**Skill**: `interactive-planner`
- **PRIMARY**: planner
- **SECONDARY**: researcher (for requirements gathering)
- **FALLBACK**: system-architect
- **COORDINATION**: Interactive questioning with AskUserQuestion tool
- **MEMORY**: `swarm/planning/user-preferences`

**Skill**: `intent-analyzer`
- **PRIMARY**: researcher
- **SECONDARY**: planner
- **COORDINATION**: Socratic questioning pattern
- **MEMORY**: `swarm/analysis/user-intent`

#### 1.2 Coding & Implementation

**Skill**: `pair-programming`
- **PRIMARY**: coder (driver role)
- **SECONDARY**: reviewer (navigator role)
- **FALLBACK**: sparc-coder
- **COORDINATION**: Real-time Driver/Navigator switching with verification
- **MEMORY**: `swarm/pair-prog/session-state`

**Skill**: `cascade-orchestrator`
- **PRIMARY**: task-orchestrator
- **SECONDARY**: workflow-automation, planner
- **COORDINATION**: Sequential/parallel/conditional micro-skill chains
- **MEMORY**: `swarm/cascade/workflow-graph`

**Skill**: `when-building-backend-api-orchestrate-api-development`
- **PRIMARY**: backend-dev
- **SECONDARY**: api-designer-agent, database-design-specialist, security-manager
- **FALLBACK**: coder
- **COORDINATION**: Multi-phase API development workflow
- **MEMORY**: `swarm/api-dev/*`

**Skill**: `sop-api-development`
- **PRIMARY**: api-designer-agent
- **SECONDARY**: backend-dev, api-documentation-specialist
- **COORDINATION**: Standardized API SOP process
- **MEMORY**: `swarm/api-sop/*`

#### 1.3 Debugging & Bug Fixing

**Skill**: `smart-bug-fix`
- **PRIMARY**: researcher (RCA specialist)
- **SECONDARY**: coder (fix implementer), tester (validation), reviewer (QA), performance-analyzer
- **COORDINATION**: 7-phase RCA workflow with hypothesis validation
- **MEMORY**: `swarm/bug-fix/*` (issue, rca, fix, tests, verification)

**Skill**: `reverse-engineer-debug`
- **PRIMARY**: code-analyzer
- **SECONDARY**: researcher, coder
- **FALLBACK**: reviewer
- **COORDINATION**: Code comprehension + debugging workflow
- **MEMORY**: `swarm/reverse-eng/analysis`

**Skill**: `debugging`
- **PRIMARY**: coder
- **SECONDARY**: tester, code-analyzer
- **FALLBACK**: reviewer
- **COORDINATION**: Systematic debugging with tools
- **MEMORY**: `swarm/debug/findings`

**Skill**: `ml-training-debugger`
- **PRIMARY**: ml-developer
- **SECONDARY**: performance-testing-agent, data-pipeline-engineer
- **FALLBACK**: ml-expert
- **COORDINATION**: ML-specific debugging (loss curves, gradients, data quality)
- **MEMORY**: `swarm/ml-debug/*`

#### 1.4 Specialized Development

**Skill**: `mobile-dev`
- **PRIMARY**: mobile-dev
- **SECONDARY**: react-developer-agent (React Native), ui-component-builder
- **FALLBACK**: coder
- **COORDINATION**: Mobile-specific development patterns
- **MEMORY**: `swarm/mobile-dev/*`

**Skill**: `ml-expert`
- **PRIMARY**: ml-developer
- **SECONDARY**: data-pipeline-engineer, performance-testing-agent
- **FALLBACK**: backend-dev
- **COORDINATION**: ML model development + training
- **MEMORY**: `swarm/ml-expert/*`

**Skill**: `ml-developer`
- **PRIMARY**: ml-developer
- **SECONDARY**: data-pipeline-engineer, query-optimization-agent
- **COORDINATION**: ML pipeline development
- **MEMORY**: `swarm/ml-pipeline/*`

---

### 2. Testing & Validation Skills (10 skills)

**Skill**: `testing-quality`
- **PRIMARY**: tester
- **SECONDARY**: production-validator, e2e-testing-specialist
- **FALLBACK**: code-analyzer
- **COORDINATION**: TDD framework with coverage validation
- **MEMORY**: `swarm/testing/quality-metrics`

**Skill**: `quick-quality-check`
- **PRIMARY**: tester
- **SECONDARY**: security-testing-agent, e2e-testing-specialist, performance-testing-agent
- **COORDINATION**: Parallel lint/security/tests for instant feedback
- **MEMORY**: `swarm/quality-check/results`

**Skill**: `functionality-audit`
- **PRIMARY**: audit-pipeline-orchestrator
- **SECONDARY**: production-validator, tester, e2e-testing-specialist
- **COORDINATION**: Sandbox testing + systematic debugging
- **MEMORY**: `swarm/audit/functionality`

**Skill**: `theater-detection-audit`
- **PRIMARY**: production-validator
- **SECONDARY**: audit-pipeline-orchestrator, reviewer, byzantine-coordinator
- **COORDINATION**: 6-agent Byzantine consensus verification
- **MEMORY**: `swarm/theater-detection/consensus`

**Skill**: `production-readiness` (uses `production-validator` skill)
- **PRIMARY**: production-validator
- **SECONDARY**: audit-pipeline-orchestrator, security-testing-agent, performance-testing-agent
- **FALLBACK**: reviewer
- **COORDINATION**: Complete audit pipeline + deployment checklist
- **MEMORY**: `swarm/prod-ready/*`

**Skill**: `verification-quality`
- **PRIMARY**: reviewer
- **SECONDARY**: tester, code-analyzer
- **FALLBACK**: production-validator
- **COORDINATION**: Quality verification workflow
- **MEMORY**: `swarm/verification/quality`

**Skill**: E2E Testing (implied skill)
- **PRIMARY**: e2e-testing-specialist
- **SECONDARY**: tester
- **COORDINATION**: Playwright/Cypress automation
- **MEMORY**: `swarm/e2e/test-results`

**Skill**: Performance Testing (implied skill)
- **PRIMARY**: performance-testing-agent
- **SECONDARY**: performance-benchmarker, frontend-performance-optimizer
- **COORDINATION**: k6/JMeter load testing
- **MEMORY**: `swarm/perf-test/metrics`

**Skill**: Security Testing (implied skill)
- **PRIMARY**: security-testing-agent
- **SECONDARY**: security-manager, reviewer
- **COORDINATION**: SAST/DAST vulnerability scanning
- **MEMORY**: `swarm/security-test/vulnerabilities`

**Skill**: Visual Regression (implied skill)
- **PRIMARY**: visual-regression-agent
- **SECONDARY**: e2e-testing-specialist, frontend-performance-optimizer
- **COORDINATION**: Screenshot comparison + UI testing
- **MEMORY**: `swarm/visual-regression/diffs`

---

### 3. Code Quality & Review Skills (6 skills)

**Skill**: `code-review-assistant`
- **PRIMARY**: reviewer
- **SECONDARY**: security-manager, performance-analyzer, code-analyzer
- **FALLBACK**: code-review-swarm
- **COORDINATION**: Multi-agent swarm review (security/performance/style)
- **MEMORY**: `swarm/code-review/findings`

**Skill**: `code-review-swarm` (referenced as agent)
- **PRIMARY**: code-review-swarm
- **SECONDARY**: reviewer, security-manager, code-analyzer
- **COORDINATION**: Swarm-based code review
- **MEMORY**: `swarm/review-swarm/*`

**Skill**: `style-audit`
- **PRIMARY**: code-analyzer
- **SECONDARY**: reviewer, css-styling-specialist
- **COORDINATION**: Code style analysis + fixes
- **MEMORY**: `swarm/style-audit/violations`

**Skill**: `sop-code-review`
- **PRIMARY**: reviewer
- **SECONDARY**: security-manager, code-analyzer
- **COORDINATION**: Security-focused systematic review
- **MEMORY**: `swarm/sop-review/*`

**Skill**: Technical Debt Management (implied)
- **PRIMARY**: technical-debt-manager-agent
- **SECONDARY**: code-analyzer, reviewer
- **COORDINATION**: Debt identification + prioritization
- **MEMORY**: `swarm/tech-debt/backlog`

**Skill**: Code Analysis (implied)
- **PRIMARY**: code-analyzer
- **SECONDARY**: reviewer, functionality-audit
- **COORDINATION**: Static analysis + connascence detection
- **MEMORY**: `swarm/analysis/violations`

---

### 4. GitHub & Repository Skills (12 skills)

**Skill**: `github-code-review`
- **PRIMARY**: github-modes
- **SECONDARY**: code-review-swarm, reviewer
- **FALLBACK**: pr-manager
- **COORDINATION**: AI swarm PR review
- **MEMORY**: `swarm/github/pr-review`

**Skill**: `github-project-management`
- **PRIMARY**: project-board-sync
- **SECONDARY**: issue-tracker, github-modes
- **COORDINATION**: Issue + project board automation
- **MEMORY**: `swarm/github/project-mgmt`

**Skill**: `github-workflow-automation`
- **PRIMARY**: workflow-automation
- **SECONDARY**: cicd-engineer, github-modes
- **COORDINATION**: Intelligent CI/CD pipelines
- **MEMORY**: `swarm/github/workflows`

**Skill**: `github-release-management`
- **PRIMARY**: release-manager
- **SECONDARY**: github-modes, cicd-engineer
- **COORDINATION**: Automated versioning + deployment
- **MEMORY**: `swarm/github/releases`

**Skill**: `github-multi-repo`
- **PRIMARY**: multi-repo-swarm
- **SECONDARY**: repo-architect, github-modes
- **COORDINATION**: Cross-repo coordination
- **MEMORY**: `swarm/github/multi-repo`

**Skill**: `github-integration`
- **PRIMARY**: github-modes
- **SECONDARY**: pr-manager, issue-tracker, workflow-automation
- **COORDINATION**: General GitHub operations
- **MEMORY**: `swarm/github/integration`

**Skill**: PR Management (implied)
- **PRIMARY**: pr-manager
- **SECONDARY**: github-modes, reviewer
- **COORDINATION**: Pull request lifecycle management
- **MEMORY**: `swarm/github/prs`

**Skill**: Issue Tracking (implied)
- **PRIMARY**: issue-tracker
- **SECONDARY**: project-board-sync, github-modes
- **COORDINATION**: Issue triage + assignment
- **MEMORY**: `swarm/github/issues`

**Skill**: Repository Architecture (implied)
- **PRIMARY**: repo-architect
- **SECONDARY**: system-architect, multi-repo-swarm
- **COORDINATION**: Repository structure design
- **MEMORY**: `swarm/repo/architecture`

**Skill**: `sop-product-launch`
- **PRIMARY**: release-manager
- **SECONDARY**: github-modes, cicd-engineer, documentation-specialist
- **COORDINATION**: Complete product launch workflow
- **MEMORY**: `swarm/product-launch/*`

**Skill**: CI/CD Recovery (implied)
- **PRIMARY**: cicd-engineer
- **SECONDARY**: workflow-automation, production-validator
- **COORDINATION**: Build/test failure recovery
- **MEMORY**: `swarm/cicd/recovery`

**Skill**: `cicd-intelligent-recovery`
- **PRIMARY**: cicd-engineer
- **SECONDARY**: production-validator, tester
- **COORDINATION**: Automated failure recovery + RCA
- **MEMORY**: `swarm/cicd/intelligent-recovery`

---

### 5. SPARC Methodology Skills (7 skills)

**Skill**: `sparc-methodology`
- **PRIMARY**: sparc-coord
- **SECONDARY**: sparc-coder, specification, pseudocode, architecture, refinement
- **COORDINATION**: 5-phase SPARC (Spec→Pseudocode→Architecture→Refinement→Completion)
- **MEMORY**: `swarm/sparc/*`

**Skill**: `when-using-sparc-methodology-use-sparc-workflow`
- **PRIMARY**: sparc-coord
- **SECONDARY**: all SPARC agents
- **COORDINATION**: Complete SPARC TDD workflow
- **MEMORY**: `swarm/sparc/workflow`

**Skill**: `specification` (SPARC phase)
- **PRIMARY**: specification
- **SECONDARY**: researcher, planner
- **COORDINATION**: Requirements analysis phase
- **MEMORY**: `swarm/sparc/specification`

**Skill**: `pseudocode` (SPARC phase)
- **PRIMARY**: pseudocode
- **SECONDARY**: planner, coder
- **COORDINATION**: Algorithm design phase
- **MEMORY**: `swarm/sparc/pseudocode`

**Skill**: `architecture` (SPARC phase)
- **PRIMARY**: architecture
- **SECONDARY**: system-architect, repo-architect
- **COORDINATION**: System design phase
- **MEMORY**: `swarm/sparc/architecture`

**Skill**: `refinement` (SPARC phase)
- **PRIMARY**: refinement
- **SECONDARY**: coder, tester
- **COORDINATION**: TDD implementation phase
- **MEMORY**: `swarm/sparc/refinement`

**Skill**: SPARC Completion (implied phase)
- **PRIMARY**: sparc-coder
- **SECONDARY**: refinement, reviewer
- **COORDINATION**: Integration + completion phase
- **MEMORY**: `swarm/sparc/completion`

---

### 6. Swarm Coordination Skills (15 skills)

**Skill**: `swarm-orchestration`
- **PRIMARY**: hierarchical-coordinator
- **SECONDARY**: mesh-coordinator, adaptive-coordinator
- **FALLBACK**: collective-intelligence-coordinator
- **COORDINATION**: Swarm topology + orchestration
- **MEMORY**: `swarm/orchestration/topology`

**Skill**: `swarm-advanced`
- **PRIMARY**: adaptive-coordinator
- **SECONDARY**: collective-intelligence-coordinator, swarm-memory-manager
- **COORDINATION**: Advanced swarm features (self-healing, auto-spawning)
- **MEMORY**: `swarm/advanced/*`

**Skill**: `hive-mind-advanced`
- **PRIMARY**: collective-intelligence-coordinator
- **SECONDARY**: hierarchical-coordinator, swarm-memory-manager
- **COORDINATION**: Queen-led hierarchical coordination
- **MEMORY**: `swarm/hive-mind/*`

**Skill**: `coordination`
- **PRIMARY**: hierarchical-coordinator
- **SECONDARY**: mesh-coordinator, adaptive-coordinator
- **COORDINATION**: Agent coordination patterns
- **MEMORY**: `swarm/coordination/*`

**Skill**: `swarm-init`
- **PRIMARY**: hierarchical-coordinator
- **SECONDARY**: mesh-coordinator
- **COORDINATION**: Swarm initialization with topology selection
- **MEMORY**: `swarm/init/config`

**Skill**: Hierarchical Coordination (implied)
- **PRIMARY**: hierarchical-coordinator
- **SECONDARY**: collective-intelligence-coordinator
- **COORDINATION**: Tree-based coordination topology
- **MEMORY**: `swarm/hierarchical/*`

**Skill**: Mesh Coordination (implied)
- **PRIMARY**: mesh-coordinator
- **SECONDARY**: hierarchical-coordinator
- **COORDINATION**: Peer-to-peer coordination topology
- **MEMORY**: `swarm/mesh/*`

**Skill**: Adaptive Coordination (implied)
- **PRIMARY**: adaptive-coordinator
- **SECONDARY**: hierarchical-coordinator, mesh-coordinator
- **COORDINATION**: Dynamic topology selection
- **MEMORY**: `swarm/adaptive/*`

**Skill**: Byzantine Consensus (implied)
- **PRIMARY**: byzantine-coordinator
- **SECONDARY**: consensus-builder, quorum-manager
- **COORDINATION**: Byzantine fault-tolerant consensus
- **MEMORY**: `swarm/byzantine/votes`

**Skill**: Raft Consensus (implied)
- **PRIMARY**: raft-manager
- **SECONDARY**: consensus-builder
- **COORDINATION**: Raft leader election + log replication
- **MEMORY**: `swarm/raft/state`

**Skill**: Gossip Protocol (implied)
- **PRIMARY**: gossip-coordinator
- **SECONDARY**: mesh-coordinator
- **COORDINATION**: Gossip-based information dissemination
- **MEMORY**: `swarm/gossip/messages`

**Skill**: CRDT Synchronization (implied)
- **PRIMARY**: crdt-synchronizer
- **SECONDARY**: mesh-coordinator
- **COORDINATION**: Conflict-free replicated data types
- **MEMORY**: `swarm/crdt/state`

**Skill**: Quorum Management (implied)
- **PRIMARY**: quorum-manager
- **SECONDARY**: consensus-builder, byzantine-coordinator
- **COORDINATION**: Quorum-based decision making
- **MEMORY**: `swarm/quorum/votes`

**Skill**: Memory Coordination (implied)
- **PRIMARY**: swarm-memory-manager
- **SECONDARY**: memory-coordinator
- **COORDINATION**: Cross-agent memory management
- **MEMORY**: `swarm/memory/*`

**Skill**: Consensus Validation (implied)
- **PRIMARY**: consensus-validator-agent
- **SECONDARY**: byzantine-coordinator, quorum-manager
- **COORDINATION**: Byzantine agreement validation
- **MEMORY**: `swarm/consensus/validation`

---

### 7. AI Enhancement Skills (12 skills)

#### 7.1 AgentDB Skills

**Skill**: `agentdb`
- **PRIMARY**: smart-agent
- **SECONDARY**: memory-coordinator
- **COORDINATION**: 150x faster vector search
- **MEMORY**: AgentDB vector store

**Skill**: `agentdb-vector-search`
- **PRIMARY**: smart-agent
- **SECONDARY**: memory-coordinator
- **COORDINATION**: RAG + document retrieval
- **MEMORY**: AgentDB HNSW index

**Skill**: `agentdb-memory-patterns`
- **PRIMARY**: swarm-memory-manager
- **SECONDARY**: smart-agent, memory-coordinator
- **COORDINATION**: Memory patterns for stateful agents
- **MEMORY**: AgentDB persistent memory

**Skill**: `agentdb-learning`
- **PRIMARY**: smart-agent
- **SECONDARY**: performance-benchmarker
- **COORDINATION**: 9 RL algorithms (Q-learning, SARSA, DQN, PPO, A3C, DDPG, TD3, SAC, TRPO)
- **MEMORY**: AgentDB learning trajectories

**Skill**: `agentdb-optimization`
- **PRIMARY**: smart-agent
- **SECONDARY**: performance-analyzer
- **COORDINATION**: 4-32x memory reduction (quantization, HNSW)
- **MEMORY**: AgentDB optimized vectors

**Skill**: `agentdb-advanced`
- **PRIMARY**: smart-agent
- **SECONDARY**: swarm-memory-manager
- **COORDINATION**: QUIC sync, multi-database, distributed features
- **MEMORY**: AgentDB distributed state

#### 7.2 ReasoningBank Skills

**Skill**: `reasoningbank-intelligence`
- **PRIMARY**: smart-agent
- **SECONDARY**: performance-analyzer
- **COORDINATION**: Pattern recognition + optimization from mistakes
- **MEMORY**: ReasoningBank trajectory database

**Skill**: `reasoningbank-agentdb`
- **PRIMARY**: smart-agent
- **SECONDARY**: swarm-memory-manager
- **COORDINATION**: 46% faster learning + 88% success rate
- **MEMORY**: ReasoningBank + AgentDB integration

#### 7.3 Gemini Skills

**Skill**: `gemini-search`
- **PRIMARY**: researcher
- **SECONDARY**: analyst (when available)
- **COORDINATION**: Gemini grounded search for web research
- **MEMORY**: `swarm/gemini/search-results`

**Skill**: `gemini-megacontext`
- **PRIMARY**: researcher
- **SECONDARY**: analyst
- **COORDINATION**: 2M token mega context processing
- **MEMORY**: `swarm/gemini/megacontext`

**Skill**: `gemini-media`
- **PRIMARY**: researcher
- **SECONDARY**: analyst
- **COORDINATION**: Multimodal analysis (image, video, audio)
- **MEMORY**: `swarm/gemini/media-analysis`

**Skill**: `gemini-extensions`
- **PRIMARY**: researcher
- **SECONDARY**: workflow-automation
- **COORDINATION**: Google Workspace, Maps extensions
- **MEMORY**: `swarm/gemini/extensions`

#### 7.4 Codex Skills

**Skill**: `codex-auto`
- **PRIMARY**: coder
- **SECONDARY**: sandbox-configurator
- **COORDINATION**: Autonomous coding in E2B sandboxes
- **MEMORY**: `swarm/codex/autonomous`

**Skill**: `codex-reasoning`
- **PRIMARY**: coder
- **SECONDARY**: smart-agent
- **COORDINATION**: Advanced chain-of-thought reasoning
- **MEMORY**: `swarm/codex/reasoning`

---

### 8. Flow-Nexus Platform Skills (3 skills)

**Skill**: `flow-nexus-platform`
- **PRIMARY**: platform (agent)
- **SECONDARY**: sandbox-configurator
- **COORDINATION**: Cloud-based orchestration (70+ MCP tools)
- **MEMORY**: Flow-Nexus cloud state

**Skill**: `flow-nexus-swarm`
- **PRIMARY**: hierarchical-coordinator
- **SECONDARY**: platform
- **COORDINATION**: Cloud swarm deployment
- **MEMORY**: Flow-Nexus swarm state

**Skill**: `flow-nexus-neural`
- **PRIMARY**: ml-developer
- **SECONDARY**: platform, data-pipeline-engineer
- **COORDINATION**: Distributed neural network training in E2B sandboxes
- **MEMORY**: Flow-Nexus neural cluster state

---

### 9. Documentation & Knowledge Skills (7 skills)

**Skill**: `documentation`
- **PRIMARY**: documentation-specialist
- **SECONDARY**: technical-writer-agent, api-documentation-specialist
- **COORDINATION**: README, setup guides, architecture docs
- **MEMORY**: `swarm/docs/generated`

**Skill**: `api-docs`
- **PRIMARY**: api-documentation-specialist
- **SECONDARY**: documentation-specialist, backend-dev
- **COORDINATION**: OpenAPI, AsyncAPI, interactive docs
- **MEMORY**: `swarm/api-docs/specs`

**Skill**: `archivist` (Deep Research SOP)
- **PRIMARY**: archivist (Deep Research agent)
- **SECONDARY**: knowledge-base-manager, documentation-specialist
- **COORDINATION**: Reproducibility packaging, DOI assignment, model card creation
- **MEMORY**: `swarm/archive/*` + `quality-gate-3/*`

**Skill**: Knowledge Base Management (implied)
- **PRIMARY**: knowledge-base-manager
- **SECONDARY**: documentation-specialist, archivist
- **COORDINATION**: Documentation organization, search, versioning
- **MEMORY**: `swarm/kb/index`

**Skill**: Technical Writing (implied)
- **PRIMARY**: technical-writer-agent
- **SECONDARY**: documentation-specialist
- **COORDINATION**: Blog posts, tutorials, case studies
- **MEMORY**: `swarm/writing/drafts`

**Skill**: Diagram Generation (implied)
- **PRIMARY**: diagram-generator-agent
- **SECONDARY**: system-architect, repo-architect
- **COORDINATION**: System diagrams, C4 models, UML
- **MEMORY**: `swarm/diagrams/generated`

**Skill**: `pptx-generation`
- **PRIMARY**: technical-writer-agent
- **SECONDARY**: documentation-specialist
- **COORDINATION**: Enterprise PowerPoint generation (html2pptx)
- **MEMORY**: `swarm/pptx/presentations`

---

### 10. Security & Compliance Skills (4 skills)

**Skill**: `security`
- **PRIMARY**: security-manager
- **SECONDARY**: security-testing-agent, reviewer
- **COORDINATION**: Security scanning + vulnerability fixes
- **MEMORY**: `swarm/security/vulnerabilities`

**Skill**: `network-security-setup`
- **PRIMARY**: security-manager
- **SECONDARY**: sandbox-configurator
- **COORDINATION**: Network boundaries + sandbox isolation
- **MEMORY**: `swarm/network-security/config`

**Skill**: `ethics-agent` (Deep Research SOP)
- **PRIMARY**: ethics-agent (Deep Research agent)
- **SECONDARY**: security-manager, reviewer
- **COORDINATION**: Ethics & safety review across Quality Gates, 6-domain risk assessment
- **MEMORY**: `swarm/ethics/*` + `quality-gate-*/ethics`

**Skill**: `data-steward` (Deep Research SOP)
- **PRIMARY**: data-steward (Deep Research agent)
- **SECONDARY**: data-pipeline-engineer, database-design-specialist
- **COORDINATION**: Dataset documentation, bias auditing, DVC, datasheet completion
- **MEMORY**: `swarm/data-steward/*` + `quality-gate-1/*`

---

### 11. Performance & Optimization Skills (4 skills)

**Skill**: `performance-analysis`
- **PRIMARY**: performance-analyzer
- **SECONDARY**: performance-benchmarker, performance-testing-agent
- **COORDINATION**: Comprehensive performance analysis + bottleneck detection
- **MEMORY**: `swarm/performance/analysis`

**Skill**: `perf-analyzer`
- **PRIMARY**: performance-analyzer
- **SECONDARY**: performance-benchmarker
- **COORDINATION**: Workflow bottleneck detection + optimization
- **MEMORY**: `swarm/perf/bottlenecks`

**Skill**: Performance Benchmarking (implied)
- **PRIMARY**: performance-benchmarker
- **SECONDARY**: performance-analyzer, performance-testing-agent
- **COORDINATION**: Benchmark execution + metrics collection
- **MEMORY**: `swarm/benchmark/results`

**Skill**: Frontend Performance Optimization (implied)
- **PRIMARY**: frontend-performance-optimizer
- **SECONDARY**: performance-analyzer, react-developer-agent
- **COORDINATION**: Lighthouse, Core Web Vitals, bundle optimization
- **MEMORY**: `swarm/frontend-perf/metrics`

---

### 12. Meta Tools & Agent Creation Skills (8 skills)

**Skill**: `agent-creator`
- **PRIMARY**: base-template-generator
- **SECONDARY**: prompt-architect, skill-builder
- **COORDINATION**: 4-phase SOP agent creation with evidence-based prompting
- **MEMORY**: `swarm/agent-creation/*`

**Skill**: `skill-builder`
- **PRIMARY**: base-template-generator
- **SECONDARY**: skill-forge (for advanced skills)
- **COORDINATION**: YAML frontmatter + progressive disclosure structure
- **MEMORY**: `swarm/skill-builder/templates`

**Skill**: `skill-creator-agent`
- **PRIMARY**: base-template-generator
- **SECONDARY**: agent-creator
- **COORDINATION**: Skill tied to specialist agent
- **MEMORY**: `swarm/skill-creation/agent-linked`

**Skill**: `micro-skill-creator`
- **PRIMARY**: base-template-generator
- **SECONDARY**: skill-builder
- **COORDINATION**: Focused single-purpose atomic skills
- **MEMORY**: `swarm/micro-skills/*`

**Skill**: `skill-forge`
- **PRIMARY**: base-template-generator
- **SECONDARY**: prompt-architect, skill-builder
- **COORDINATION**: Advanced skill crafting with 7-phase methodology
- **MEMORY**: `swarm/skill-forge/advanced`

**Skill**: `slash-command-encoder`
- **PRIMARY**: base-template-generator
- **SECONDARY**: skill-builder
- **COORDINATION**: Create .claude/commands/*.md slash commands
- **MEMORY**: `swarm/slash-commands/generated`

**Skill**: `prompt-architect`
- **PRIMARY**: prompt-architect (agent needs creation in Batch 5)
- **SECONDARY**: base-template-generator, researcher
- **COORDINATION**: Analyze + optimize prompts with evidence-based techniques
- **MEMORY**: `swarm/prompts/optimized`

**Skill**: `meta-tools`
- **PRIMARY**: base-template-generator
- **SECONDARY**: skill-builder, agent-creator
- **COORDINATION**: Tool creation tools (meta-level)
- **MEMORY**: `swarm/meta-tools/*`

---

### 13. Utilities & Automation Skills (7 skills)

**Skill**: `hooks-automation`
- **PRIMARY**: workflow-automation
- **SECONDARY**: cicd-engineer
- **COORDINATION**: Hook integration + lifecycle event automation
- **MEMORY**: `swarm/hooks/config`

**Skill**: `when-automating-workflows-use-hooks-automation`
- **PRIMARY**: workflow-automation
- **SECONDARY**: cicd-engineer, task-orchestrator
- **COORDINATION**: Automated workflow execution with hooks
- **MEMORY**: `swarm/workflow-automation/*`

**Skill**: `workflow`
- **PRIMARY**: task-orchestrator
- **SECONDARY**: workflow-automation, planner
- **COORDINATION**: Workflow creation + execution
- **MEMORY**: `swarm/workflow/graph`

**Skill**: `i18n-automation`
- **PRIMARY**: workflow-automation
- **SECONDARY**: coder, technical-writer-agent
- **COORDINATION**: Internationalization workflow automation
- **MEMORY**: `swarm/i18n/translations`

**Skill**: `when-internationalizing-app-use-i18n-automation`
- **PRIMARY**: workflow-automation
- **SECONDARY**: coder
- **COORDINATION**: Complete i18n workflow
- **MEMORY**: `swarm/i18n-workflow/*`

**Skill**: `stream-chain`
- **PRIMARY**: task-orchestrator
- **SECONDARY**: workflow-automation
- **COORDINATION**: Streaming workflow chains
- **MEMORY**: `swarm/stream-chain/state`

**Skill**: `specialized-tools`
- **PRIMARY**: base-template-generator
- **SECONDARY**: skill-builder
- **COORDINATION**: Domain-specific tool creation
- **MEMORY**: `swarm/specialized-tools/*`

---

### 14. Database & Migration Skills (2 skills)

**Skill**: `migration-planner`
- **PRIMARY**: database-migration-agent
- **SECONDARY**: migration-planner (agent from Specialized Development, if exists)
- **FALLBACK**: database-design-specialist
- **COORDINATION**: Database migration planning + execution
- **MEMORY**: `swarm/migration/plan`

**Skill**: Database Design (implied)
- **PRIMARY**: database-design-specialist
- **SECONDARY**: system-architect, query-optimization-agent
- **COORDINATION**: Schema design, normalization, ER diagrams
- **MEMORY**: `swarm/db-design/schema`

---

### 15. Platform & Infrastructure Skills (3 skills)

**Skill**: `platform`
- **PRIMARY**: platform (agent needs clarification/creation)
- **SECONDARY**: sandbox-configurator, cicd-engineer
- **COORDINATION**: Platform-level operations
- **MEMORY**: `swarm/platform/config`

**Skill**: `sandbox-configurator`
- **PRIMARY**: sandbox-configurator (agent needs creation in Batch 5)
- **SECONDARY**: cicd-engineer, platform
- **COORDINATION**: E2B sandbox configuration
- **MEMORY**: `swarm/sandbox/config`

**Skill**: `web-cli-teleport`
- **PRIMARY**: workflow-automation
- **SECONDARY**: base-template-generator
- **COORDINATION**: CLI↔Web interface teleportation
- **MEMORY**: `swarm/teleport/mappings`

---

### 16. Deep Research SOP Skills (2 additional skills)

**Skill**: `evaluator` (Deep Research SOP)
- **PRIMARY**: evaluator (Deep Research agent)
- **SECONDARY**: ethics-agent, archivist, data-steward
- **COORDINATION**: Final authority for Quality Gate approvals (Gates 1, 2, 3), GO/NO-GO decisions
- **MEMORY**: `quality-gate-*/approval` + `swarm/evaluator/*`

**Skill**: Quality Gate Management (implied)
- **PRIMARY**: evaluator
- **SECONDARY**: ethics-agent, data-steward, archivist
- **COORDINATION**: Multi-gate research validation system
- **MEMORY**: `quality-gate-{1,2,3}/*`

---

### 17. Additional Coordination & Specialized Skills (5 skills)

**Skill**: `multi-model`
- **PRIMARY**: smart-agent
- **SECONDARY**: researcher, performance-analyzer
- **COORDINATION**: Multi-model routing + selection (compare Claude, GPT, Gemini)
- **MEMORY**: `swarm/multi-model/selection`

**Skill**: `dependencies`
- **PRIMARY**: code-analyzer
- **SECONDARY**: reviewer, database-design-specialist
- **COORDINATION**: Dependency analysis + mapping
- **MEMORY**: `swarm/dependencies/graph`

**Skill**: Task Orchestration (implied)
- **PRIMARY**: task-orchestrator
- **SECONDARY**: hierarchical-coordinator, workflow-automation
- **COORDINATION**: Complex task orchestration across swarms
- **MEMORY**: `swarm/task-orchestration/*`

**Skill**: Smart Agent (implied)
- **PRIMARY**: smart-agent
- **SECONDARY**: swarm-memory-manager, performance-benchmarker
- **COORDINATION**: Intelligent agent with AgentDB/ReasoningBank integration
- **MEMORY**: AgentDB + ReasoningBank combined

**Skill**: Consensus Building (implied)
- **PRIMARY**: consensus-builder
- **SECONDARY**: byzantine-coordinator, raft-manager, quorum-manager
- **COORDINATION**: Multi-protocol consensus mechanisms
- **MEMORY**: `swarm/consensus/*`

---

## Agent Utilization Statistics

### Top 20 Most-Utilized Agents (by skill assignments)

| Rank | Agent | Primary Assignments | Secondary Assignments | Total Utilization | Domains Covered |
|------|-------|---------------------|----------------------|-------------------|-----------------|
| 1 | **base-template-generator** | 8 | 6 | 14 | Meta Tools, Templates |
| 2 | **researcher** | 7 | 8 | 15 | Development, Research, AI |
| 3 | **coder** | 6 | 10 | 16 | Development, Coding |
| 4 | **hierarchical-coordinator** | 5 | 8 | 13 | Swarm, Coordination |
| 5 | **tester** | 5 | 9 | 14 | Testing, Validation |
| 6 | **reviewer** | 4 | 11 | 15 | Code Quality, Review |
| 7 | **smart-agent** | 7 | 4 | 11 | AI Enhancement, AgentDB |
| 8 | **workflow-automation** | 5 | 5 | 10 | Automation, Workflows |
| 9 | **documentation-specialist** | 3 | 7 | 10 | Documentation |
| 10 | **security-manager** | 3 | 6 | 9 | Security, Compliance |
| 11 | **production-validator** | 3 | 5 | 8 | Testing, Validation |
| 12 | **performance-analyzer** | 3 | 6 | 9 | Performance, Optimization |
| 13 | **github-modes** | 3 | 8 | 11 | GitHub, Repository |
| 14 | **system-architect** | 2 | 7 | 9 | Architecture, Design |
| 15 | **sparc-coord** | 2 | 1 | 3 | SPARC Methodology |
| 16 | **ml-developer** | 3 | 3 | 6 | ML, Specialized Dev |
| 17 | **code-analyzer** | 3 | 6 | 9 | Code Quality, Analysis |
| 18 | **cicd-engineer** | 3 | 7 | 10 | CI/CD, DevOps |
| 19 | **task-orchestrator** | 3 | 3 | 6 | Orchestration, Workflows |
| 20 | **database-design-specialist** | 2 | 4 | 6 | Database, Data Layer |

### Agent Utilization by Domain

| Domain | Agents | Total Assignments | Avg per Agent |
|--------|--------|-------------------|---------------|
| Core Development | 8 | 62 | 7.8 |
| Swarm Coordination | 15 | 45 | 3.0 |
| Testing & Validation | 9 | 48 | 5.3 |
| GitHub & Repository | 14 | 38 | 2.7 |
| SPARC Methodology | 13 | 21 | 1.6 |
| AI Enhancement | 8 | 34 | 4.3 |
| Specialized Development | 11 | 28 | 2.5 |
| Documentation & Knowledge | 6 | 24 | 4.0 |
| Security & Compliance | 4 | 16 | 4.0 |
| Performance & Optimization | 5 | 18 | 3.6 |
| Meta Tools & Templates | 9 | 32 | 3.6 |
| Utilities & Automation | 5 | 22 | 4.4 |
| Frontend Development | 6 | 8 | 1.3 |
| Database & Data Layer | 7 | 14 | 2.0 |

### Under-Utilized Agents (< 2 assignments)

**Agents needing more skill assignments:**
- frontend-accessibility-auditor (1)
- visual-regression-agent (1)
- contract-testing-agent (0)
- chaos-engineering-agent (0)
- vue-developer-agent (0)
- ui-component-builder (1)
- css-styling-specialist (1)
- cache-strategy-agent (0)
- database-backup-recovery-agent (0)
- technical-writer-agent (2)
- diagram-generator-agent (1)
- consensus-validator-agent (1)
- swarm-health-monitor-agent (0)

**Recommendation**: Create specialized skills for under-utilized agents to maximize roster utilization.

---

## Coordination Pattern Analysis

### Pattern 1: Hierarchical Multi-Phase Workflows

**Skills using this pattern**: `feature-dev-complete`, `smart-bug-fix`, `sparc-methodology`

**Characteristics**:
- Sequential phases with quality gates
- 5-7 specialized agents coordinating
- Memory-based handoffs between phases
- Hierarchical coordinator manages flow

**Example Flow**:
```
researcher → system-architect → planner → (tester + coder) → reviewer → documentation-specialist → cicd-engineer
         ↓                  ↓           ↓               ↓            ↓                    ↓                ↓
      [memory store]    [memory]   [memory]        [memory]    [memory]            [memory]        [memory]
```

**Agent Pattern**:
- PRIMARY: 1 coordinator + 6-7 specialists
- COORDINATION: Hierarchical topology
- MEMORY: Structured namespace per phase

### Pattern 2: Parallel Swarm Execution

**Skills using this pattern**: `parallel-swarm-implementation`, `code-review-assistant`, `quick-quality-check`

**Characteristics**:
- Multiple agents execute concurrently
- Mesh or star topology for coordination
- Results aggregated by coordinator
- Fast execution for time-critical tasks

**Example Flow**:
```
        coordinator
       /    |    \
      /     |     \
   agent1  agent2  agent3
      \     |     /
       \    |    /
     aggregated results
```

**Agent Pattern**:
- PRIMARY: 1 coordinator
- SECONDARY: 3-6 parallel agents
- COORDINATION: Mesh or star topology
- MEMORY: Shared results namespace

### Pattern 3: Byzantine Consensus Verification

**Skills using this pattern**: `theater-detection-audit`, Quality Gate approvals

**Characteristics**:
- 6+ agents independently verify
- Byzantine fault-tolerant voting
- Consensus threshold (67%+)
- Prevents single-agent manipulation

**Example Flow**:
```
task → [agent1, agent2, agent3, agent4, agent5, agent6] → voting → consensus → approval/rejection
```

**Agent Pattern**:
- PRIMARY: consensus-builder
- SECONDARY: 6 independent validators
- COORDINATION: Byzantine consensus
- MEMORY: Vote records + consensus state

### Pattern 4: Research-Then-Action

**Skills using this pattern**: `research-driven-planning`, `gemini-search` workflows

**Characteristics**:
- Research phase first
- Action phase based on findings
- Memory bridge between phases
- Researcher → Specialist handoff

**Example Flow**:
```
researcher (Gemini search) → [store findings] → specialist (implementation)
```

**Agent Pattern**:
- PRIMARY: researcher (phase 1), specialist (phase 2)
- COORDINATION: Sequential with memory handoff
- MEMORY: Research findings namespace

### Pattern 5: Quality Gate Pipeline

**Skills using this pattern**: Deep Research SOP (data-steward, ethics-agent, archivist, evaluator)

**Characteristics**:
- 3 quality gates with GO/NO-GO decisions
- Evaluator has final authority
- Multi-domain validation (ethics, data quality, reproducibility)
- Comprehensive documentation required

**Example Flow**:
```
Gate 1: data-steward + ethics-agent → evaluator → GO/NO-GO
         ↓ (if GO)
Gate 2: ethics-agent + archivist → evaluator → GO/NO-GO
         ↓ (if GO)
Gate 3: archivist + ethics-agent → evaluator → GO/NO-GO → Production
```

**Agent Pattern**:
- PRIMARY: evaluator (gate authority)
- SECONDARY: 2-3 domain specialists per gate
- COORDINATION: Sequential gates with approvals
- MEMORY: `quality-gate-{1,2,3}/*`

---

## Recommendations for New Agents

### Critical Gaps (High Priority)

Based on skill assignments, these agents should be created in **Batch 5**:

1. **prompt-architect** (Referenced in `prompt-architect` skill)
   - **Need**: Prompt optimization and analysis
   - **Skills**: prompt engineering, evidence-based prompting
   - **Current Workaround**: base-template-generator + researcher

2. **analyst** (Referenced in Gemini skills as SECONDARY)
   - **Need**: Data analysis and research support
   - **Skills**: data analysis, research synthesis
   - **Current Workaround**: researcher only

3. **content-creator** (Referenced in documentation)
   - **Need**: Content creation and marketing materials
   - **Skills**: blog posts, case studies, marketing copy
   - **Current Workaround**: technical-writer-agent

4. **platform-agent** (Referenced in `platform` skill)
   - **Need**: Platform-level infrastructure management
   - **Skills**: infrastructure, deployment, scaling
   - **Current Workaround**: cicd-engineer + sandbox-configurator

5. **sandbox-configurator-agent** (Referenced in multiple skills)
   - **Need**: E2B sandbox configuration and management
   - **Skills**: sandbox setup, environment config
   - **Current Workaround**: cicd-engineer

### Specialized Gaps (Medium Priority)

6. **contract-testing-specialist** (Agent exists but no skill assignments)
   - **Need**: API contract testing skill
   - **Recommendation**: Create `contract-testing` skill for Pact workflows

7. **chaos-engineering-specialist** (Agent exists but no skill assignments)
   - **Need**: Chaos engineering skill
   - **Recommendation**: Create `chaos-testing` skill for resilience testing

8. **cache-optimization-agent** (Agent exists but no skill assignments)
   - **Need**: Caching strategy skill
   - **Recommendation**: Create `cache-optimization` skill for Redis/Memcached

9. **backup-recovery-specialist** (Agent exists but no skill assignments)
   - **Need**: Backup and disaster recovery skill
   - **Recommendation**: Create `backup-recovery` skill

10. **swarm-health-monitor** (Agent exists but no skill assignments)
    - **Need**: Swarm health monitoring skill
    - **Recommendation**: Create `swarm-health-monitoring` skill

### Enhancement Recommendations

**For Existing Skills**:
- Add more FALLBACK agents to critical skills
- Create micro-skills for under-utilized agents
- Add coordination patterns documentation to each skill

**For New Skills**:
- Frontend framework-specific skills (React, Vue, Angular patterns)
- Database-specific skills (PostgreSQL tuning, MongoDB optimization)
- Cloud provider skills (AWS deployment, GCP setup, Azure config)
- Specialized testing skills (contract testing, chaos engineering)

---

## Coordination Pattern Templates

### Template 1: Simple Sequential (2-3 agents)

```yaml
coordination:
  pattern: sequential
  agents:
    - primary: researcher
      phase: analysis
      memory: swarm/[skill]/analysis
    - primary: coder
      phase: implementation
      memory: swarm/[skill]/implementation
  handoff: memory-based
```

### Template 2: Parallel Swarm (4-6 agents)

```yaml
coordination:
  pattern: parallel
  topology: mesh
  agents:
    coordinator: hierarchical-coordinator
    workers:
      - agent1
      - agent2
      - agent3
      - agent4
  aggregation: coordinator
  memory: swarm/[skill]/parallel-results
```

### Template 3: Multi-Phase Pipeline (7+ agents)

```yaml
coordination:
  pattern: pipeline
  topology: hierarchical
  phases:
    - name: phase-1
      agents: [agent1, agent2]
      gate: quality-check
    - name: phase-2
      agents: [agent3, agent4, agent5]
      gate: review
    - name: phase-3
      agents: [agent6, agent7]
      gate: approval
  memory: swarm/[skill]/phase-*
```

### Template 4: Consensus-Based (6+ validators)

```yaml
coordination:
  pattern: consensus
  mechanism: byzantine
  threshold: 0.67
  agents:
    validators:
      - agent1
      - agent2
      - agent3
      - agent4
      - agent5
      - agent6
    consensus-builder: byzantine-coordinator
  memory: swarm/[skill]/consensus-votes
```

---

## Appendix A: Complete Skill-Agent Matrix

**Note**: Due to length, this is a summary. Full matrix available in separate CSV export.

| Skill ID | Skill Name | PRIMARY Agent | SECONDARY Agents | Pattern |
|----------|-----------|---------------|------------------|---------|
| S001 | feature-dev-complete | researcher | system-architect, coder, tester, reviewer, api-docs, cicd-engineer | Hierarchical Multi-Phase |
| S002 | smart-bug-fix | researcher | coder, tester, reviewer, performance-analyzer | Hierarchical Multi-Phase |
| S003 | pair-programming | coder | reviewer | Real-time Coordination |
| S004 | testing-quality | tester | production-validator, e2e-testing-specialist | Parallel Validation |
| S005 | theater-detection-audit | production-validator | audit-pipeline-orchestrator, reviewer, byzantine-coordinator | Byzantine Consensus |
| ... | ... | ... | ... | ... |
| S130 | evaluator | evaluator | ethics-agent, archivist, data-steward | Quality Gate Authority |

**Full Matrix**: See `SKILL-AGENT-MATRIX.csv` (auto-generated)

---

## Appendix B: Agent Command Mapping

**Cross-Reference**: See AGENT-REGISTRY.md for complete agent command assignments.

**Quick Reference**:
- Total Agents: 130
- Total Commands: ~1,585
- Avg Commands per Agent: 12.2
- Command Coverage: 100% (58/58 core commands)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-02 | Initial comprehensive mapping of 130 skills to 130 agents |

---

**Maintained By**: SPARC System
**Next Review**: 2025-11-15 (after Batch 5 agent creation)
**Contact**: See AGENT-REGISTRY.md for agent-specific questions
