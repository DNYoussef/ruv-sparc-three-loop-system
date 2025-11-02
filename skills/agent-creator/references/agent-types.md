# Agent Types & Patterns

## Overview

Agents can be categorized into three primary types based on their scope and capabilities:

1. **Specialist Agents**: Deep expertise in single domain
2. **Coordinator Agents**: Multi-agent orchestration, workflow management
3. **Hybrid Agents**: Multi-domain expertise + coordination capabilities

---

## Type 1: Specialist Agents

### Characteristics

- **Single Domain Focus**: Expert in one specific area (marketing, frontend dev, data analysis)
- **Deep Knowledge**: Comprehensive understanding of domain tools, patterns, best practices
- **Autonomous Execution**: Can complete domain-specific tasks independently
- **Limited Coordination**: Rarely spawns sub-agents (self-sufficient within domain)

### When to Use

Use specialist agents for:
- ✅ Tasks requiring deep domain expertise
- ✅ Recurring workflows within single domain
- ✅ When speed matters (no coordination overhead)
- ✅ When consistency is critical (domain-specific best practices)

### Agent Structure

```markdown
# [SPECIALIST NAME] AGENT - SYSTEM PROMPT

## 🎭 CORE IDENTITY
I am a **[Specialist Title]** with deep expertise in [domain].
- [Expertise Area 1]
- [Expertise Area 2]
- [Expertise Area 3]

## 📋 UNIVERSAL COMMANDS
[Standard file, git, memory operations]

## 🎯 MY SPECIALIST COMMANDS
[Domain-specific commands, 5-10 commands]

## 🧠 COGNITIVE FRAMEWORK
[Evidence-based techniques: self-consistency, PoT, plan-and-solve]

## 🚧 GUARDRAILS
[Domain-specific failure modes, 3-5 guardrails]

## ✅ SUCCESS CRITERIA
[Domain-specific completion criteria]

## 📖 WORKFLOW EXAMPLES
[2-3 complete workflows with exact commands]
```

### Examples

**Marketing Specialist**:
- Domain: Marketing strategy, campaign optimization, audience analytics
- Tools: Google Analytics, SEMrush, HubSpot
- Commands: `/campaign-analyze`, `/audience-segment`, `/ab-test-design`, `/roi-calculate`
- Coordination: Minimal (delegates to Data Analyst for complex stats only)

**Frontend Developer Specialist**:
- Domain: React, TypeScript, UI/UX, accessibility
- Tools: React 18, Vite, TailwindCSS, Vitest
- Commands: `/react-component`, `/frontend-optimize`, `/accessibility-audit`
- Coordination: None (fully autonomous within frontend domain)

**Database Specialist**:
- Domain: PostgreSQL, query optimization, schema design
- Tools: PostgreSQL 15, Prisma, pgAdmin, EXPLAIN ANALYZE
- Commands: `/db-schema`, `/db-migration`, `/db-optimize-query`, `/db-backup`
- Coordination: Minimal (coordinates with Backend Developer for API contracts)

### Creation Time

- **First-time**: 3-4 hours (Phase 1-4 complete)
- **Speed-run**: 1.5-2 hours (experienced creators with templates)

---

## Type 2: Coordinator Agents

### Characteristics

- **Orchestration Focus**: Manages workflows, spawns/coordinates multiple specialist agents
- **Broad Knowledge**: Understanding of multiple domains (not deep expertise in any)
- **Dependency Management**: Resolves task dependencies, builds execution DAGs
- **Error Recovery**: Handles failures, rollbacks, escalations
- **State Management**: Tracks deployment state, resource inventories

### When to Use

Use coordinator agents for:
- ✅ Multi-agent workflows (5+ agents)
- ✅ Complex dependencies (sequential, parallel, conditional execution)
- ✅ Infrastructure orchestration (DevOps, CI/CD)
- ✅ When error recovery is critical (rollback capabilities)

### Agent Structure

```markdown
# [COORDINATOR NAME] AGENT - SYSTEM PROMPT

## 🎭 CORE IDENTITY
I am a **[Coordinator Title]** specializing in multi-agent orchestration for [workflow type].

**Coordination Expertise**:
- [Orchestration Pattern 1]
- [Orchestration Pattern 2]
- [Error Recovery]
- [State Management]

## 📋 UNIVERSAL COMMANDS
[Standard file, git, memory, AGENT COORDINATION]

## 🎯 MY COORDINATOR COMMANDS
[Orchestration-specific commands]
- /orchestrate-[workflow]
- /build-dag
- /spawn-agents-wave
- /check-agent-health
- /rollback-[workflow]

## 🔧 MCP SERVER TOOLS I USE
[Claude Flow MCP emphasis - swarm_init, agent_spawn, task_orchestrate]

## 🧠 COGNITIVE FRAMEWORK
### Dependency Resolution
[DAG construction, cycle detection]

### Error Propagation
[Detect, classify, retry/rollback, notify]

### Parallel Execution
[Wave-based execution, rate limiting]

## 🚧 GUARDRAILS
[Coordinator-specific failure modes]
- Never deploy without rollback manifest
- Never proceed with DAG cycles
- Never spawn unlimited agents

## ✅ SUCCESS CRITERIA
[Coordination-specific completion criteria]
- All agents completed successfully
- No orphaned resources
- Rollback tested

## 📖 WORKFLOW EXAMPLES
[2-3 orchestration workflows with agent spawning]
```

### Examples

**DevOps Coordinator**:
- Orchestrates: Terraform, Kubernetes, Docker, Monitoring agents
- Workflows: Infrastructure deployment, rollback, health monitoring
- Key Patterns: DAG resolution, parallel execution, error recovery
- State: Deployment manifests, resource IDs, rollback state

**Project Manager Coordinator**:
- Orchestrates: Developer, Designer, Tester, Reviewer agents
- Workflows: Feature development, sprint planning, release management
- Key Patterns: Task decomposition, dependency tracking, progress monitoring
- State: Project timeline, task assignments, completion status

**CI/CD Pipeline Coordinator**:
- Orchestrates: Build, Test, Security Scan, Deploy agents
- Workflows: Continuous integration, continuous deployment, rollback
- Key Patterns: Pipeline stages, failure handling, artifact management
- State: Build artifacts, test results, deployment history

### Creation Time

- **First-time**: 4-5 hours (orchestration logic complex)
- **Speed-run**: 2.5-3 hours (with DAG templates)

---

## Type 3: Hybrid Agents

### Characteristics

- **Multi-Domain Expertise**: Deep knowledge in 2-4 related domains
- **Coordinator Capabilities**: Can spawn sub-agents for complex sub-tasks
- **Context Switching**: Efficiently transitions between domains
- **End-to-End Ownership**: Completes features spanning multiple domains
- **Intelligent Delegation**: Knows when to delegate vs. implement directly

### When to Use

Use hybrid agents for:
- ✅ Full-stack features (frontend + backend + database)
- ✅ Features with moderate complexity (standard auth, CRUD, forms)
- ✅ When end-to-end ownership needed (one agent, complete delivery)
- ✅ Projects with related technology stacks (React + Node.js, for example)

### Agent Structure

```markdown
# [HYBRID NAME] AGENT - SYSTEM PROMPT

## 🎭 CORE IDENTITY
I am a **[Hybrid Title]** with deep expertise across [domains] and coordinator capabilities.

**Specialist Expertise**:
- [Domain 1] - [specific skills]
- [Domain 2] - [specific skills]
- [Domain 3] - [specific skills]
- [Domain 4] - [specific skills]

**Coordinator Capabilities**:
- [Orchestration capability 1]
- [Agent delegation]
- [Quality assurance]

## 📋 UNIVERSAL COMMANDS
[Standard file, git, memory, agent coordination]

## 🎯 MY SPECIALIST COMMANDS (Multi-Domain)
### [Domain 1] Commands
[Domain 1 specific commands]

### [Domain 2] Commands
[Domain 2 specific commands]

### [Domain 3] Commands
[Domain 3 specific commands]

### Coordinator Commands
[Orchestration commands]

## 🧠 COGNITIVE FRAMEWORK (Hybrid-Specific)
### Domain Switching Protocol
[How to transition between domains]

### Delegation Decision Framework
[When to delegate vs. implement]

### [Domain]-First Workflow
[Standard end-to-end workflow]

## 🚧 GUARDRAILS (Hybrid-Specific)
[Hybrid failure modes]
- Never implement [Domain A] before [Domain B]
- Never delegate standard tasks
- Never skip [integration step]

## ✅ SUCCESS CRITERIA (Hybrid Agent)
[Multi-domain completion criteria]
- [Domain 1] complete
- [Domain 2] complete
- [Domain 3] complete
- Integration tests pass
- Deployment successful

## 📖 WORKFLOW EXAMPLES
### Workflow 1: Standard Feature (Hybrid Solo)
[End-to-end implementation without delegation]

### Workflow 2: Complex Feature (Hybrid + Delegation)
[Hybrid coordinates specialists for complex sub-tasks]
```

### Examples

**Full-Stack Developer**:
- Domains: Frontend (React/TS), Backend (Node.js), Database (PostgreSQL), DevOps (Docker)
- Coordination: Spawns Algorithm, Security, Performance specialists for complex sub-tasks
- Workflow: API-first design → Backend → Frontend → Tests → Deploy
- Delegation: Delegates when complexity >2 hours OR outside 4 core domains

**Data Scientist + ML Engineer**:
- Domains: Data analysis (Python/pandas), ML modeling (scikit-learn/PyTorch), ML ops (MLflow), Deployment (FastAPI)
- Coordination: Spawns Data Engineer for ETL pipelines, DevOps for Kubernetes deployment
- Workflow: Data exploration → Feature engineering → Model training → Deployment → Monitoring
- Delegation: Delegates infrastructure, complex ETL

**Mobile + Backend Developer**:
- Domains: Mobile (React Native), Backend (Node.js), Database (PostgreSQL), API design
- Coordination: Spawns UI Designer for complex animations, Security Specialist for auth
- Workflow: API design → Backend → Mobile UI → Integration tests → App store deployment
- Delegation: Delegates UI polish, security audits

### Creation Time

- **First-time**: 5-6 hours (multiple domains + coordination logic)
- **Speed-run**: 3-3.5 hours (with multi-domain templates)

---

## Decision Matrix: Which Agent Type?

### Use Specialist When:

| Criteria | Specialist | Coordinator | Hybrid |
|----------|-----------|-------------|--------|
| **Task Scope** | Single domain | Multi-agent workflow | Multi-domain feature |
| **Depth Needed** | Expert-level | Broad understanding | Expert in 2-4 domains |
| **Coordination** | Minimal | High | Moderate |
| **Speed** | Fastest (no coordination overhead) | Slower (coordination) | Medium |
| **Complexity** | Domain-specific | Orchestration logic | Multi-domain + delegation |
| **Example Tasks** | "Optimize PostgreSQL queries" | "Deploy full-stack app" | "Build auth feature (frontend + backend)" |

### Use Coordinator When:

| Criteria | Value |
|----------|-------|
| **Number of Agents** | 5+ specialist agents |
| **Dependencies** | Complex (DAG, sequential, parallel) |
| **Error Recovery** | Critical (rollback required) |
| **State Management** | Complex (resource tracking, inventory) |
| **Example Tasks** | "Orchestrate infrastructure deployment", "Manage CI/CD pipeline", "Coordinate sprint workflow" |

### Use Hybrid When:

| Criteria | Value |
|----------|-------|
| **Number of Domains** | 2-4 related domains |
| **Feature Scope** | End-to-end (crosses domains) |
| **Ownership Model** | Single agent, complete delivery |
| **Delegation Frequency** | Occasional (10-20% of tasks) |
| **Example Tasks** | "Build user authentication (frontend + backend + DB)", "Create recommendation engine (data + ML + API)" |

---

## Specialization Patterns by Domain

### Analytical Agents (Specialist)

**Focus**: Data analysis, evidence evaluation, validation

**Key Characteristics**:
- Self-consistency validation (multi-angle checks)
- Statistical rigor (significance tests, confidence intervals)
- Data quality standards (completeness, accuracy)

**Examples**: Data Analyst, Research Analyst, Quality Auditor, A/B Test Analyst

### Generative Agents (Specialist)

**Focus**: Content creation, design, synthesis

**Key Characteristics**:
- Plan-and-solve execution (outline → draft → polish)
- Quality criteria (readability, engagement, accuracy)
- Refinement cycles (iterative improvement)

**Examples**: Content Writer, Marketing Copywriter, UI Designer, Documentation Specialist

### Diagnostic Agents (Specialist)

**Focus**: Problem identification, debugging, root cause analysis

**Key Characteristics**:
- Program-of-thought decomposition (break problem into sub-problems)
- Hypothesis generation and testing
- Systematic troubleshooting

**Examples**: Debugging Specialist, Performance Analyzer, Security Auditor, Incident Responder

### Orchestration Agents (Coordinator)

**Focus**: Workflow coordination, agent management, dependency resolution

**Key Characteristics**:
- DAG construction and cycle detection
- Parallel execution with rate limiting
- Error propagation and rollback

**Examples**: DevOps Coordinator, Project Manager, CI/CD Pipeline Coordinator, Workflow Orchestrator

### Multi-Domain Agents (Hybrid)

**Focus**: End-to-end feature delivery across multiple domains

**Key Characteristics**:
- Context switching protocol
- Delegation decision framework
- Integration-first workflows

**Examples**: Full-Stack Developer, Data Scientist + ML Engineer, Mobile + Backend Developer

---

## Mixing Patterns: When to Combine

### Specialist + Coordinator

**Scenario**: Specialist agent occasionally coordinates with other specialists

**Example**: Security Specialist agent that:
- Performs security audits (specialist)
- Spawns Penetration Testing agent for advanced testing (coordinator)

**Implementation**:
```markdown
## When to Coordinate
I spawn sub-agents when:
- Penetration testing required (delegate to Penetration Testing Specialist)
- Complex cryptographic analysis (delegate to Cryptography Specialist)

Otherwise, I work autonomously within security domain.
```

### Hybrid + Extensive Coordination

**Scenario**: Hybrid agent frequently delegates, acts more like coordinator

**Example**: Technical Lead agent that:
- Has expertise in architecture, backend, frontend (hybrid)
- Frequently spawns Developer agents for implementation (coordinator)
- Focuses on high-level design, code review, delegation

**Implementation**:
```markdown
## Delegation Strategy
I delegate most implementation tasks:
- Backend implementation → Backend Developer
- Frontend implementation → Frontend Developer
- Database design → Database Specialist

I focus on:
- Architecture design
- API contract definition
- Code review
- Integration oversight
```

**Note**: This is close to pure coordinator with domain expertise (useful for tech leads, architects)

---

## Summary

**Agent Types**:
1. **Specialist**: Deep single-domain expertise, autonomous, minimal coordination
2. **Coordinator**: Multi-agent orchestration, broad knowledge, error recovery
3. **Hybrid**: Multi-domain expertise + coordination, end-to-end ownership

**Decision Criteria**:
- **Task scope**: Single domain → Specialist | Multi-agent → Coordinator | Multi-domain feature → Hybrid
- **Depth needed**: Expert-level → Specialist | Broad understanding → Coordinator | Expert in 2-4 domains → Hybrid
- **Coordination**: Minimal → Specialist | High → Coordinator | Moderate → Hybrid

**Creation Complexity**:
- Specialist: 3-4 hours (simplest)
- Coordinator: 4-5 hours (orchestration logic)
- Hybrid: 5-6 hours (most complex - multiple domains + coordination)

**Choose wisely**: Match agent type to task requirements for optimal performance.
