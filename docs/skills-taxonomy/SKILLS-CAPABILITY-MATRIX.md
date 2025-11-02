# Skills Capability Matrix - Agent-Command-Tool Mapping

**Version:** 3.0.0  
**Last Updated:** 2025-11-02  
**Purpose**: Map skills to agents, commands, and MCP tools for capability discovery

---

## Matrix Overview

This document provides comprehensive capability mapping showing:
1. **Which agents support which skills**
2. **Which commands trigger which skills**
3. **Which MCP tools each skill requires**
4. **Skill dependencies and relationships**
5. **Agent specialization matrix**

---

## Part 1: Agent-to-Skill Mapping

### By Agent Type (130 agents total)

#### Development Agents (15 agents)

| Agent | Skills Supported | Agent Type | Capabilities |
|-------|------------------|-----------|--------------|
| `coder` | 18+ development skills | Core Dev | Implementation, coding, architecture |
| `reviewer` | 8+ review skills | Core Dev | Code review, quality validation |
| `tester` | 12+ testing skills | Core Dev | Test design, execution, coverage |
| `planner` | 6+ planning skills | Core Dev | Feature planning, requirements |
| `researcher` | 14+ research skills | Core Dev | Research, literature, investigation |
| `backend-dev` | 10+ backend skills | Specialized | REST API, databases, infrastructure |
| `frontend-dev` | 8+ frontend skills | Specialized | UI/UX, component design, accessibility |
| `mobile-dev` | 6+ mobile skills | Specialized | React Native, iOS, Android |
| `ml-developer` | 10+ ML skills | Specialized | ML pipelines, training, optimization |
| `database-architect` | 4+ database skills | Specialized | Schema design, optimization, migration |
| `security-engineer` | 8+ security skills | Specialized | Security audit, penetration testing |
| `devops-engineer` | 10+ devops skills | Specialized | CI/CD, infrastructure, automation |
| `api-designer` | 6+ API skills | Specialized | API design, documentation, standards |
| `data-scientist` | 8+ data skills | Specialized | Data analysis, ML, statistics |
| `architect` | 12+ architecture skills | Specialized | System design, patterns, scalability |

#### Quality & Validation Agents (14 agents)

| Agent | Skills Supported | Agent Type | Capabilities |
|-------|------------------|-----------|--------------|
| `functionality-auditor` | functionality-audit | QA | Execution testing, verification |
| `theater-detector` | theater-detection-audit | QA | Implementation verification, fake code detection |
| `security-auditor` | security | Security | Security scanning, OWASP/SANS/CWE |
| `performance-analyzer` | performance-analysis | QA | Bottleneck detection, optimization |
| `test-engineer` | testing, testing-quality | QA | Test generation, coverage analysis |
| `code-quality-checker` | verification-quality | QA | Multi-layer quality verification |
| `style-enforcer` | style-audit | QA | Linting, formatting, consistency |
| `production-validator` | production-readiness | QA | Pre-deployment checklist |
| `reproducibility-auditor` | reproducibility-audit | QA | Reproducibility verification |
| `gate-validator` | gate-validation | QA | Quality Gate approval |
| `holistic-evaluator` | holistic-evaluation | QA | Multi-dimensional evaluation |
| `functionality-auditor` | functionality-audit | QA | Code execution testing |
| `connoisseur` | quick-quality-check | QA | Fast parallel quality checks |
| `analyzer` | dependencies | QA | Dependency analysis, circular detection |

#### Swarm & Coordination Agents (18 agents)

| Agent | Skills Supported | Agent Type | Capabilities |
|-------|------------------|-----------|--------------|
| `hierarchical-coordinator` | swarm-orchestration | Coordinator | Tree topology coordination |
| `mesh-coordinator` | swarm-advanced | Coordinator | Peer-to-peer mesh coordination |
| `adaptive-coordinator` | swarm-advanced | Coordinator | Adaptive topology selection |
| `collective-intelligence-coordinator` | hive-mind-advanced | Coordinator | Queen-led consensus |
| `swarm-memory-manager` | coordination | Coordinator | Persistent memory management |
| `task-orchestrator` | cascade-orchestrator | Coordinator | Multi-skill pipeline orchestration |
| `stream-coordinator` | stream-chain | Coordinator | Real-time workflow streaming |
| `consensus-builder` | hive-mind-advanced | Coordinator | Byzantine consensus |
| `crdt-synchronizer` | coordination | Coordinator | Conflict-free replicated data types |
| `quorum-manager` | coordination | Coordinator | Quorum-based coordination |
| `byzantine-coordinator` | hive-mind-advanced | Coordinator | Byzantine fault tolerance |
| `raft-manager` | coordination | Coordinator | Raft consensus protocol |
| `gossip-coordinator` | coordination | Coordinator | Gossip protocol messaging |
| `distributed-state-manager` | coordination | Coordinator | Distributed state synchronization |
| `workflow-manager` | workflow | Coordinator | Workflow creation and execution |
| `event-processor` | workflow | Coordinator | Event-driven processing |
| `state-synchronizer` | coordination | Coordinator | State sync across agents |
| `topology-manager` | swarm-advanced | Coordinator | Topology optimization |

#### Research & Analysis Agents (12 agents)

| Agent | Skills Supported | Agent Type | Capabilities |
|-------|------------------|-----------|--------------|
| `researcher` | research-driven-planning | Research | Evidence-based research |
| `literature-reviewer` | literature-synthesis | Research | Literature review, synthesis |
| `publication-specialist` | research-publication | Research | Paper, dataset, code publication |
| `deep-researcher` | deep-research-orchestrator | Research | Multi-stage deep research |
| `hypothesis-tester` | smart-bug-fix, debugging | Analysis | Hypothesis testing, validation |
| `root-cause-analyzer` | cicd-intelligent-recovery | Analysis | RCA methodology |
| `pattern-recognizer` | reasoningbank-intelligence | Analysis | Pattern recognition, learning |
| `evaluator` | holistic-evaluation | Analysis | Comprehensive evaluation |
| `data-steward` | reproducibility-audit | Data | Dataset quality, bias audit |
| `ethics-agent` | security, reproducibility-audit | Ethics | Ethics review, risk assessment |
| `archivist` | research-publication | Data | Artifact archival, DOI assignment |
| `reverse-engineer` | reverse-engineering-deep | Analysis | Code/firmware reverse engineering |

#### Security & Infrastructure Agents (16 agents)

| Agent | Skills Supported | Agent Type | Capabilities |
|-------|------------------|-----------|--------------|
| `security-architect` | security | Security | Security design, architecture |
| `owasp-scanner` | security | Security | OWASP top 10 scanning |
| `secrets-detective` | security | Security | Secrets and credentials detection |
| `crypto-auditor` | security | Security | Cryptography audit, standards |
| `auth-specialist` | security | Security | Authentication/authorization |
| `compliance-checker` | security, gate-validation | Compliance | Regulatory compliance |
| `penetration-tester` | security | Security | Penetration testing |
| `network-administrator` | network-security-setup | Infra | Network isolation, firewall rules |
| `sandbox-manager` | sandbox-configurator | Infra | Sandbox configuration, isolation |
| `devops-engineer` | cicd-intelligent-recovery | Infra | CI/CD pipeline management |
| `release-manager` | github-release-management | Infra | Versioning, deployment |
| `infrastructure-architect` | platform | Infra | Infrastructure design |
| `monitoring-specialist` | performance-analysis | Infra | Performance monitoring |
| `automation-engineer` | hooks-automation | Infra | Workflow automation |
| `cloud-architect` | flow-nexus-platform | Cloud | Cloud infrastructure design |
| `disaster-recovery-specialist` | cicd-intelligent-recovery | Infra | Recovery and rollback |

#### ML & Data Agents (12 agents)

| Agent | Skills Supported | Agent Type | Capabilities |
|-------|------------------|-----------|--------------|
| `data-scientist` | ml-expert | ML | Data science, analysis |
| `model-architect` | ml-expert | ML | Neural network design |
| `trainer` | ml-expert, flow-nexus-neural | ML | Model training, optimization |
| `evaluator` | ml-expert | ML | Model evaluation, metrics |
| `loss-debugger` | ml-training-debugger | ML | Training loss debugging |
| `hyperparameter-tuner` | ml-training-debugger | ML | Hyperparameter optimization |
| `data-validator` | ml-training-debugger | ML | Data validation, quality |
| `embedding-specialist` | agentdb-vector-search | ML | Embedding generation, vectors |
| `search-engineer` | agentdb-vector-search | ML | Search index optimization |
| `retrieval-optimizer` | agentdb-vector-search | ML | RAG and retrieval optimization |
| `neural-architect` | flow-nexus-neural | ML | Distributed neural design |
| `trajectory-tracker` | reasoningbank-agentdb | ML | Learning trajectory tracking |

#### Meta & Prompt Agents (13 agents)

| Agent | Skills Supported | Agent Type | Capabilities |
|-------|------------------|-----------|--------------|
| `cognitive-architect` | agent-creator | Meta | Agent design and architecture |
| `prompt-engineer` | skill-creator-agent, prompt-architect | Meta | Prompt engineering, optimization |
| `validation-specialist` | agent-creator | Meta | Evidence-based validation |
| `integration-designer` | agent-creator | Meta | System integration design |
| `documentation-writer` | agent-creator | Meta | Documentation generation |
| `skill-designer` | skill-builder | Meta | Skill structure design |
| `yaml-generator` | skill-builder | Meta | YAML frontmatter generation |
| `structure-validator` | skill-builder | Meta | Structure validation |
| `skill-architect` | skill-forge | Meta | Advanced skill engineering |
| `pattern-applier` | prompt-architect | Meta | Pattern application and testing |
| `anti-pattern-detector` | prompt-architect | Meta | Anti-pattern detection |
| `intent-mapper` | intent-analyzer | Meta | Intent analysis and mapping |
| `gap-identifier` | skill-gap-analyzer | Meta | Gap identification in skills |

#### GitHub & Integration Agents (14 agents)

| Agent | Skills Supported | Agent Type | Capabilities |
|-------|------------------|-----------|--------------|
| `pr-analyzer` | github-code-review | GitHub | PR analysis and parsing |
| `branch-manager` | github-integration | GitHub | Branch management |
| `commit-analyzer` | github-integration | GitHub | Commit analysis |
| `issue-triager` | github-project-management | GitHub | Issue triage and categorization |
| `board-manager` | github-project-management | GitHub | Project board management |
| `sprint-planner` | github-project-management | GitHub | Sprint planning and tracking |
| `progress-tracker` | github-project-management | GitHub | Progress tracking, metrics |
| `version-manager` | github-release-management | GitHub | Version management, semver |
| `release-tester` | github-release-management | GitHub | Release testing, validation |
| `deployer` | github-release-management | GitHub | Deployment orchestration |
| `rollback-handler` | github-release-management | GitHub | Rollback and recovery |
| `workflow-designer` | github-workflow-automation | GitHub | GitHub Actions design |
| `pipeline-builder` | github-workflow-automation | GitHub | CI/CD pipeline creation |
| `repo-coordinator` | github-multi-repo | GitHub | Multi-repo coordination |

#### Content & Utility Agents (12 agents)

| Agent | Skills Supported | Agent Type | Capabilities |
|-------|------------------|-----------|--------------|
| `content-planner` | pptx-generation | Content | Content planning, outline |
| `slide-designer` | pptx-generation | Content | Slide design, layout |
| `layout-optimizer` | pptx-generation | Content | Layout optimization |
| `exporter` | pptx-generation | Content | File export, formatting |
| `code-analyzer` | documentation | Content | Code analysis for docs |
| `doc-writer` | documentation | Content | Documentation writing |
| `example-generator` | documentation | Content | Code example generation |
| `diagram-creator` | documentation | Content | Diagram and visual creation |
| `translator` | i18n-automation | Utility | Translation management |
| `key-generator` | i18n-automation | Utility | i18n key generation |
| `library-integrator` | i18n-automation | Utility | Library integration |
| `validator` | i18n-automation | Utility | Validation and testing |

#### Debugging & Testing Agents (14 agents)

| Agent | Skills Supported | Agent Type | Capabilities |
|-------|------------------|-----------|--------------|
| `bug-reproducer` | debugging | Debug | Bug reproduction, isolation |
| `stack-trace-analyst` | debugging | Debug | Stack trace analysis |
| `state-inspector` | debugging | Debug | State inspection, variables |
| `hypothesis-tester` | debugging | Debug | Hypothesis testing |
| `fix-validator` | debugging | Debug | Fix validation, verification |
| `rca-analyst` | smart-bug-fix | Debug | Root cause analysis |
| `codex-iterator` | smart-bug-fix | Debug | Codex-based code generation |
| `test-validator` | smart-bug-fix | Debug | Test validation |
| `test-planner` | testing | Test | Test strategy planning |
| `unit-test-writer` | testing | Test | Unit test generation |
| `integration-test-writer` | testing | Test | Integration test generation |
| `e2e-test-writer` | testing | Test | End-to-end test generation |
| `visual-regression-tester` | testing | Test | Visual regression testing |
| `coverage-analyzer` | testing | Test | Coverage analysis |

#### Advanced Agents (14 agents)

| Agent | Skills Supported | Agent Type | Capabilities |
|-------|------------------|-----------|--------------|
| `smart-agent` | multiple | Specialized | Multi-skill coordination |
| `perf-analyzer` | performance-analysis | Analysis | Performance analysis |
| `performance-benchmarker` | performance | Analysis | Performance benchmarking |
| `memory-coordinator` | coordinate | Memory | Memory management |
| `neural-pattern-trainer` | flow-nexus-neural | Neural | Neural pattern training |
| `distributed-coordinator` | flow-nexus-swarm | Distributed | Distributed system coordination |
| `consensus-engine` | hive-mind-advanced | Consensus | Consensus mechanism |
| `vector-db-optimizer` | agentdb-optimization | Vector | Vector database optimization |
| `index-optimizer` | agentdb-optimization | Vector | Index optimization, HNSW |
| `quantization-specialist` | agentdb-optimization | Vector | Quantization and compression |
| `policy-designer` | agentdb-learning | RL | RL policy design |
| `reward-modeler` | agentdb-learning | RL | Reward function modeling |
| `rl-engineer` | agentdb-learning | RL | Reinforcement learning engineer |
| `memory-architect` | agentdb-memory-patterns | Memory | Memory architecture design |

---

## Part 2: Command-to-Skill Mapping

### Core Commands (40+ available)

#### Development Commands
```bash
# Feature Development
/feature-develop              # Complete feature development workflow
/pair-program                # Pair programming mode
/implement-api              # REST API development

# Code Generation
/generate-code              # Code generation with AI
/refactor-code              # Code refactoring
/implement-pattern          # Design pattern implementation

# Documentation
/generate-docs              # Auto-generate documentation
/generate-api-docs          # API documentation generation
```

#### Testing & Quality Commands
```bash
# Testing
/run-tests                  # Execute test suite
/generate-tests             # Test case generation
/coverage-report            # Coverage analysis report

# Quality Validation
/quick-check                # Quick quality check (5-10 min)
/detect-theater             # Theater detection audit
/validate-functionality     # Functionality validation
/verify-quality             # Quality verification
/production-ready           # Production readiness check
```

#### Analysis & Debugging Commands
```bash
# Debugging
/debug-code                 # Interactive code debugging
/debug-ml                   # ML training debugging
/analyze-performance        # Performance analysis

# Root Cause
/find-root-cause            # RCA analysis
/analyze-error              # Error analysis

# Reverse Engineering
/reverse-engineer           # Quick reverse engineering
/deep-reverse-engineer      # Deep reverse engineering
/analyze-firmware           # Firmware analysis
```

#### Research & Planning Commands
```bash
# Research
/research-topic             # Web research with Gemini
/synthesize-literature      # Literature synthesis
/deep-research              # Multi-stage deep research
/publish-research           # Research publication

# Planning
/plan-feature               # Interactive planning
/analyze-intent             # Intent analysis
/pre-mortem                 # Pre-mortem analysis
```

#### Infrastructure & Deployment Commands
```bash
# Security
/audit-security             # Security audit
/setup-network-security     # Network isolation setup
/configure-sandbox          # Sandbox configuration

# DevOps
/setup-ci-cd                # CI/CD pipeline setup
/automate-workflows         # Workflow automation
/release-software           # Release management
/deploy-to-production       # Production deployment

# GitHub
/review-pr                  # PR code review
/manage-github-project      # Project management
/automate-github            # GitHub automation
/release-github             # GitHub release
/sync-repos                 # Multi-repo synchronization
```

#### Swarm & Coordination Commands
```bash
# Orchestration
/orchestrate-swarm          # Swarm orchestration
/build-cascade              # Cascade orchestrator
/coordinate-hive            # Hive mind coordination
/stream-workflow            # Workflow streaming

# Commands
/create-slash-command       # Create ergonomic slash commands
```

#### ML & Vector Commands
```bash
# Neural Networks
/train-neural-net           # Neural network training
/deploy-swarm               # Swarm deployment
/manage-platform            # Flow Nexus management

# Vector Search
/semantic-search            # Vector semantic search
/optimize-vectors           # Vector optimization
/train-rl-agents            # RL agent training
/implement-memory           # Persistent memory
```

#### Meta Commands
```bash
# Agents & Skills
/create-agent               # Agent creation
/create-skill               # Skill creation
/create-micro-skill         # Micro-skill creation
/engineer-skill             # Advanced skill engineering
/analyze-skill-gaps         # Skill gap analysis

# Optimization
/optimize-prompt            # Prompt optimization
/architect-prompt           # Prompt engineering
/manage-tokens              # Token budget management
```

#### Content & Utility Commands
```bash
# Content
/generate-presentation      # PowerPoint generation
/generate-content           # Content creation
/translate-app              # i18n automation

# Tools
/teleport-cli               # CLI workflow transfer
/multi-model                # Multi-model routing
```

---

## Part 3: MCP Tools-to-Skill Mapping

### MCP Tool Categories Used by Skills

#### Claude Flow MCP Tools (Primary)

| MCP Tool | Skills Using | Purpose |
|----------|--------------|---------|
| `swarm_init` | swarm-orchestration, swarm-advanced | Initialize swarm topology |
| `agent_spawn` | All coordination skills | Create specialized agents |
| `task_orchestrate` | cascade-orchestrator, parallel-swarm | Orchestrate tasks |
| `swarm_status` | swarm-orchestration, swarm-advanced | Monitor swarm status |
| `agent_list` | coordination | List active agents |
| `agent_metrics` | performance-analysis | Get agent metrics |
| `task_status` | cascade-orchestrator, stream-chain | Monitor task progress |
| `benchmark_run` | performance-analysis | Run performance benchmarks |
| `features_detect` | platform | Detect runtime features |
| `memory_usage` | agentdb-optimization | Monitor memory |
| `neural_status` | flow-nexus-neural | Neural network status |
| `neural_train` | flow-nexus-neural, ml-expert | Train neural networks |

#### Flow Nexus Cloud MCP Tools (Secondary)

| MCP Tool | Skills Using | Purpose |
|----------|--------------|---------|
| `swarm_init` | flow-nexus-swarm | Cloud swarm initialization |
| `sandbox_create` | flow-nexus-platform, functionality-audit | Create E2B sandboxes |
| `sandbox_execute` | flow-nexus-platform, functionality-audit | Execute code in sandbox |
| `neural_train` | flow-nexus-neural | Distributed neural training |
| `neural_cluster_init` | flow-nexus-neural | Initialize neural cluster |
| `neural_node_deploy` | flow-nexus-neural | Deploy neural nodes |
| `template_list` | flow-nexus-platform | List project templates |
| `template_deploy` | flow-nexus-platform | Deploy from template |

#### AgentDB MCP Tools (Integration)

| MCP Tool | Skills Using | Purpose |
|----------|--------------|---------|
| `vector_search` | agentdb-vector-search, reasoningbank-agentdb | Semantic vector search |
| `memory_store` | agentdb-memory-patterns, all swarm skills | Persistent memory storage |
| `daa_init` | coordination, agentdb-learning | Initialize DAA system |
| `daa_agent_create` | agent-creator, skill-creator-agent | Create autonomous agents |
| `daa_workflow_create` | cascade-orchestrator, workflow | Create autonomous workflows |
| `daa_workflow_execute` | cascade-orchestrator, workflow | Execute workflows |

#### GitHub MCP Tools (Specialized)

| MCP Tool | Skills Using | Purpose |
|----------|--------------|---------|
| `github_repo_analyze` | github-integration | Repository analysis |
| `github_code_review` | github-code-review | PR code review |
| `github_project_management` | github-project-management | Project board management |
| `github_release_management` | github-release-management | Release automation |
| `github_workflow_automation` | github-workflow-automation | CI/CD automation |
| `github_multi_repo` | github-multi-repo | Multi-repo coordination |

#### Memory MCP Tools (Cross-Session)

| MCP Tool | Skills Using | Purpose |
|----------|--------------|---------|
| `vector_search` | All skills with memory persistence | Semantic memory retrieval |
| `memory_store` | All skills | Persistent cross-session memory |

#### Connascence Analyzer MCP Tools (Code Quality)

| MCP Tool | Skills Using | Purpose |
|----------|--------------|---------|
| `analyze_file` | verification-quality, code-review-assistant | Single file analysis |
| `analyze_workspace` | holistic-evaluation | Workspace-wide analysis |
| `health_check` | deployment-readiness | System health verification |

---

## Part 4: Skill Dependency & Relationship Matrix

### Dependency Chains

**Feature Development Chain**:
```
research-driven-planning
    ↓
interactive-planner (or intent-analyzer)
    ↓
feature-dev-complete
    ├→ pair-programming
    ├→ i18n-automation (if i18n needed)
    └→ documentation
    ↓
testing
    ↓
functionality-audit
    ↓
theater-detection-audit
    ↓
production-readiness
    ↓
github-release-management (deployment)
```

**Security-Focused Chain**:
```
security (comprehensive audit)
    ↓
network-security-setup
    ↓
sandbox-configurator
    ↓
code-review-assistant
    ↓
verification-quality
    ↓
production-readiness
```

**ML Development Chain**:
```
research-driven-planning
    ↓
ml-expert (model development)
    ├→ agentdb-learning (if using RL)
    ├→ agentdb-vector-search (if using vectors)
    └→ flow-nexus-neural (if distributed)
    ↓
ml-training-debugger (if issues)
    ↓
baseline-replication
    ↓
reproducibility-audit
    ↓
research-publication (if publishing)
```

**Swarm Orchestration Chain**:
```
interactive-planner
    ↓
swarm-orchestration (or swarm-advanced)
    ├→ cascade-orchestrator (multi-skill)
    ├→ stream-chain (real-time)
    └→ hive-mind-advanced (consensus)
    ↓
performance-analysis
    ↓
github-workflow-automation (if GitHub)
    ↓
flow-nexus-swarm (if cloud deployment)
```

### Skill Complementarity Matrix

#### High Affinity Pairs (typically used together)

| Skill 1 | Skill 2 | Reason |
|---------|---------|--------|
| testing | functionality-audit | Verify test execution in sandbox |
| theater-detection-audit | verification-quality | Comprehensive validation |
| security | code-review-assistant | Security-focused code review |
| ml-expert | ml-training-debugger | Debug training issues |
| research-driven-planning | feature-dev-complete | Plan before implementation |
| github-code-review | code-review-assistant | AI-enhanced PR review |
| agentdb-vector-search | agentdb-optimization | Optimize semantic search |
| swarm-orchestration | cascade-orchestrator | Orchestrate multi-skill pipelines |
| performance-analysis | reverse-engineering-deep | Understand perf bottlenecks |
| pptx-generation | documentation | Create presentation docs |

#### Complementary Triplets (often used together)

```
(testing, functionality-audit, theater-detection-audit)
  → Complete code validation pipeline

(security, network-security-setup, sandbox-configurator)
  → Complete sandbox security setup

(research-driven-planning, feature-dev-complete, production-readiness)
  → Complete feature lifecycle

(swarm-orchestration, cascade-orchestrator, stream-chain)
  → Advanced workflow orchestration

(ml-expert, agentdb-learning, flow-nexus-neural)
  → Complete ML pipeline
```

---

## Part 5: Agent Specialization Matrix

### By Development Phase

#### Planning Phase Agents
- researcher (investigation, research)
- planner (requirement gathering)
- intent-analyzer (disambiguation)
- interactive-planner (structured questioning)

#### Design Phase Agents
- architect (system design)
- skill-architect (skill design)
- api-designer (API design)
- database-architect (schema design)

#### Implementation Phase Agents
- coder (general coding)
- backend-dev (backend implementation)
- frontend-dev (frontend implementation)
- mobile-dev (mobile implementation)
- ml-developer (ML implementation)

#### Testing Phase Agents
- tester (test generation)
- test-engineer (test orchestration)
- qa-engineer (quality assurance)
- functionality-auditor (execution testing)

#### Validation Phase Agents
- reviewer (code review)
- security-auditor (security validation)
- performance-analyzer (performance validation)
- code-quality-checker (quality validation)
- theater-detector (implementation verification)

#### Deployment Phase Agents
- devops-engineer (CI/CD setup)
- release-manager (release orchestration)
- cloud-architect (cloud deployment)
- production-validator (production readiness)

### By Domain/Specialization

#### Backend/API Development
- backend-dev (implementation)
- database-architect (data layer)
- api-designer (API design)
- security-engineer (security)
- performance-analyzer (optimization)

#### Frontend Development
- frontend-dev (implementation)
- ui-designer (UI design)
- performance-analyzer (optimization)
- accessibility-specialist (a11y)
- visual-regression-tester (testing)

#### Data/ML Development
- data-scientist (data analysis)
- ml-developer (ML implementation)
- neural-architect (neural design)
- trainer (model training)
- loss-debugger (training debugging)

#### Infrastructure/DevOps
- devops-engineer (CI/CD)
- cloud-architect (cloud design)
- security-engineer (infrastructure security)
- network-administrator (networking)
- monitoring-specialist (observability)

#### Research/Academia
- researcher (research conduct)
- literature-reviewer (literature review)
- publication-specialist (publication)
- data-steward (data management)
- ethics-agent (ethics review)

---

## Part 6: Tool Requirements by Skill

### Skills Requiring Zero External Tools

```
interactive-planner        (Question-based)
intent-analyzer           (LLM-based analysis)
pair-programming          (AI-assisted coding)
style-audit               (Linting rules)
documentation             (Template-based)
i18n-automation           (Configuration)
pptx-generation           (Template-based)
web-cli-teleport          (Context transfer)
multi-model               (Model routing)
```

### Skills Requiring 1-2 MCP Tools

```
functionality-audit       → sandbox_create, sandbox_execute
theater-detection-audit   → sandbox_execute
testing                   → sandbox_execute
debugging                 → sandbox_execute
quick-quality-check       → sandbox_execute
ml-training-debugger      → sandbox_execute
performance               → benchmark_run
agentdb-vector-search     → vector_search, memory_store
agentdb-memory-patterns   → memory_store
skill-builder             → memory_store
```

### Skills Requiring 3-5 MCP Tools

```
feature-dev-complete      → swarm_init, agent_spawn, task_orchestrate, memory_store, sandbox_execute
code-review-assistant     → swarm_init, agent_spawn, memory_store, github_code_review
security                  → sandbox_execute, memory_store, connascence_analyze
cicd-intelligent-recovery → github_workflow_automation, sandbox_execute, memory_store
research-driven-planning  → memory_store, swarm_init, agent_spawn
```

### Skills Requiring 5+ MCP Tools

```
parallel-swarm-implementation  → swarm_init, agent_spawn, task_orchestrate, 
                                 memory_store, sandbox_execute, github integration

flow-nexus-platform            → swarm_init, sandbox_create, sandbox_execute,
                                 neural_train, template_deploy, memory_store

flow-nexus-neural              → neural_cluster_init, neural_node_deploy,
                                 neural_train, sandbox_execute, memory_store

swarm-orchestration            → swarm_init, agent_spawn, task_orchestrate,
                                 swarm_status, agent_metrics, memory_store

github-multi-repo              → github_repo_analyze, github_multi_repo,
                                 memory_store, swarm_init, agent_spawn
```

---

## Part 7: Performance Characteristics Matrix

### Execution Time Distribution

```
< 5 minutes (Fast Utilities):
  - interactive-planner
  - intent-analyzer
  - network-security-setup
  - sandbox-configurator
  - slash-command-encoder
  - multi-model
  - web-cli-teleport

5-15 minutes (Quick Operations):
  - quick-quality-check
  - theater-detection-audit
  - style-audit
  - gemini-search
  - reverse-engineering-quick
  - token-budget-advisor
  - skill-gap-analyzer

15-60 minutes (Standard Tasks):
  - documentation
  - i18n-automation
  - hooks-automation
  - dependencies
  - pair-programming (initial)
  - deployment-readiness
  - prompt-architect
  - prompt-optimization-analyzer
  - method-development

1-4 hours (Complex Tasks):
  - feature-dev-complete
  - code-review-assistant
  - when-building-backend-api-orchestrate-api-development
  - testing
  - testing-quality
  - functionality-audit
  - security
  - production-readiness
  - ml-training-debugger
  - performance-analysis
  - debugging
  - smart-bug-fix
  - reverse-engineering-deep
  - reverse-engineering-firmware
  - github-code-review
  - github-workflow-automation
  - github-release-management
  - cascade-orchestrator
  - swarm-orchestration
  - swarm-advanced
  - hive-mind-advanced
  - stream-chain
  - flow-nexus-platform
  - flow-nexus-neural
  - flow-nexus-swarm
  - ml-expert
  - reasoningbank-intelligence
  - pptx-generation

4+ hours (Extended Projects):
  - when-releasing-new-product-orchestrate-product-launch (8 weeks)
  - when-reviewing-pull-request-orchestrate-comprehensive-code-review (4 hours)
  - sop-product-launch (8 weeks)
  - sop-code-review (4 hours)
  - sop-api-development (2 weeks)
  - research-driven-planning (6-11 hours)
  - literature-synthesis (8-16 hours)
  - deep-research-orchestrator (12-24 hours)
  - research-publication (8-16 hours)
  - parallel-swarm-implementation (4-6 hours)
  - cicd-intelligent-recovery (1.5-2 hours for execution)
```

### Resource Usage Pattern

```
High Resource (10+ agents, 5+ MCP tools):
  - parallel-swarm-implementation
  - feature-dev-complete
  - flow-nexus-platform
  - flow-nexus-neural
  - swarm-orchestration
  - deep-research-orchestrator
  - research-publication
  - sop-product-launch
  - when-releasing-new-product-orchestrate-product-launch

Medium Resource (5-10 agents, 2-5 MCP tools):
  - code-review-assistant
  - github-code-review
  - github-multi-repo
  - security
  - ml-expert
  - testing
  - cicd-intelligent-recovery
  - cascade-orchestrator
  - stream-chain
  - research-driven-planning

Low Resource (1-5 agents, 0-2 MCP tools):
  - documentation
  - style-audit
  - debugging
  - quick-quality-check
  - theater-detection-audit
  - pair-programming
  - prompt-architect
  - skill-builder
  - agent-creator
  - interactive-planner
```

### Token Efficiency

```
Ultra-Efficient (<1K tokens):
  - interactive-planner
  - intent-analyzer
  - multi-model
  - web-cli-teleport

Efficient (1-10K tokens):
  - style-audit
  - dependencies
  - quick-quality-check
  - skill-gap-analyzer
  - token-budget-advisor

Moderate (10-100K tokens):
  - feature-dev-complete
  - code-review-assistant
  - testing
  - documentation
  - pair-programming
  - debugging

High (100K+ tokens):
  - research-driven-planning
  - literature-synthesis
  - deep-research-orchestrator
  - parallel-swarm-implementation
  - sop-product-launch
  - flow-nexus-neural
```

---

## Part 8: Recommended Skill Combinations

### Common Workflows

#### "Ship Fast" (1 week)
```
1. interactive-planner (plan)
2. feature-dev-complete (build)
3. quick-quality-check (verify)
4. github-release-management (deploy)
Total Time: 2-4 days
```

#### "Enterprise Quality" (2-4 weeks)
```
1. research-driven-planning (research + pre-mortem)
2. when-building-backend-api-orchestrate-api-development (structured)
3. testing + functionality-audit (thorough testing)
4. security + code-review-assistant (comprehensive review)
5. production-readiness (pre-deployment)
6. github-release-management (deploy)
Total Time: 2-4 weeks
```

#### "Research & Publish" (2-3 months)
```
1. research-driven-planning (kick-off)
2. literature-synthesis (background)
3. deep-research-orchestrator (primary research)
4. ml-expert (if applicable) + baseline-replication
5. reproducibility-audit (validate)
6. research-publication (publish)
Total Time: 2-3 months
```

#### "CI/CD Excellence" (1-2 weeks)
```
1. sparc-methodology (architecture)
2. github-workflow-automation (pipeline setup)
3. cicd-intelligent-recovery (intelligent recovery)
4. github-release-management (release automation)
5. monitoring/performance-analysis (ongoing)
Total Time: 1-2 weeks
```

#### "Swarm Orchestration" (2-4 hours)
```
1. interactive-planner (define requirements)
2. swarm-orchestration (or swarm-advanced)
3. cascade-orchestrator (multi-skill pipeline)
4. performance-analysis (optimization)
5. stream-chain (real-time monitoring)
Total Time: 2-4 hours
```

---

## Summary Statistics

### Coverage Metrics

- **Total Skills**: 96
- **Total Agents**: 130
- **Total Commands**: 40+
- **MCP Tools Used**: 40+
- **Avg Agents per Skill**: 4-6
- **Avg Tools per Skill**: 1-3
- **High-Complexity Skills**: 60 (62.5%)
- **Medium-Complexity Skills**: 28 (29.2%)
- **Low-Complexity Skills**: 8 (8.3%)

### Agent Utilization

- **Most-Used Agents**: coder, reviewer, tester (80+ skills)
- **Specialized Agents**: ml-developer, mobile-dev (5-10 skills each)
- **Coordinator Agents**: 18 agents serving swarm skills
- **Support Agents**: 40+ agents in specialized roles

### Skill Relationships

- **Strongly Dependent Pairs**: 25+
- **Complementary Triplets**: 15+
- **Isolated Skills**: 3-5 (minimal dependencies)
- **Dependency Chains**: 8+ major chains

---

**Version**: 3.0.0  
**Last Updated**: 2025-11-02  
**Maintenance**: Claude Code Development Team
