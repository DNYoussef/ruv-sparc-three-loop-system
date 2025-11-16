# Enhanced Playbook System - Complete Skill Sequences

**Generated**: 2025-11-14
**Total Skills**: 122
**Total Playbooks**: 25+
**Categories**: 10

---

## Table of Contents

1. [Universal Workflow](#universal-workflow)
2. [Delivery Playbooks](#delivery-playbooks) (5 playbooks)
3. [Operations Playbooks](#operations-playbooks) (4 playbooks)
4. [Research Playbooks](#research-playbooks) (4 playbooks)
5. [Security Playbooks](#security-playbooks) (3 playbooks)
6. [Quality Playbooks](#quality-playbooks) (3 playbooks)
7. [Platform Playbooks](#platform-playbooks) (3 playbooks)
8. [GitHub Playbooks](#github-playbooks) (3 playbooks)
9. [Three-Loop Playbook](#three-loop-playbook) (FLAGSHIP)
10. [Deep Research Playbook](#deep-research-playbook) (FLAGSHIP)
11. [Specialist Playbooks](#specialist-playbooks) (4 playbooks)

---

## Universal Workflow

**Every request flows through this pattern:**

```mermaid
User Request
    ↓
🔍 intent-analyzer (Auto-triggered on ambiguous/complex requests)
    ├─ Probabilistic intent mapping (>80% confidence = proceed)
    ├─ First principles decomposition
    ├─ Constraint detection (explicit + implicit)
    └─ Socratic clarification if needed (<80% confidence)
    ↓
✨ prompt-architect (Auto-triggered after intent analysis)
    ├─ Optimize prompt for clarity
    ├─ Add missing context
    ├─ Structure request for optimal response
    └─ Evidence-based prompting patterns
    ↓
🎯 cascade-orchestrator OR deep-research-orchestrator
    ├─ Select playbook based on intent type
    ├─ Route to optimal skill sequence
    ├─ Multi-model routing (Gemini/Codex/Claude)
    └─ Coordinate parallel execution via swarm
    ↓
⚡ Playbook Execution (skill sequence with actual skills)
```

**Key Insight**: You don't manually select skills - the system auto-routes!

---

## Delivery Playbooks

### 1. Simple Feature Implementation

**When**: "Add password reset feature", "Build REST API endpoint"

**Sequence**:
```yaml
1. intent-analyzer (if ambiguous)
2. prompt-architect (optimize requirements)
3. sparc-methodology
   Phase 1: Specification
   Phase 2: Pseudocode
   Phase 3: Architecture
   Phase 4: Refinement (TDD)
   Phase 5: Code
4. quick-quality-check (parallel validation)
5. functionality-audit (sandbox testing)
6. sop-code-review
```

**Skills Used**: 7 skills
**Time**: 2-4 hours
**Complexity**: Low

---

### 2. Complex Feature (Three-Loop) 🔥 FLAGSHIP

**When**: "Add payment processing", "Build auth system", "Implement data pipeline"

**Sequence**:
```yaml
# Loop 1: Research-Driven Planning (2-4 hours)
1. intent-analyzer
2. prompt-architect
3. research-driven-planning
   - Gemini search for best practices
   - 5x pre-mortem cycles
   - Multi-agent consensus
   - Technology selection
   - Output: Validated plan (<3% failure confidence)

# Loop 2: Parallel Swarm Implementation (4-8 hours)
4. parallel-swarm-implementation
   - Spawn 6-10 specialist agents in parallel
   - Dynamic agent+skill execution graphs
   - Theater detection via Byzantine consensus
   - Agents: researcher, coder, reviewer, tester, documenter, security-analyst
   - Output: Reality-validated implementation

5. theater-detection-audit
   - 6-agent Byzantine consensus
   - Sandbox execution testing
   - Real implementation verification

# Loop 3: CI/CD Intelligent Recovery (1-2 hours)
6. cicd-intelligent-recovery
   - Automated test execution
   - Root cause analysis
   - Automated repair and re-validation
   - Output: 100% test success, production-ready

7. production-readiness
   - Complete audit pipeline
   - Deployment checklist
   - Security validation
```

**Skills Used**: 7 core skills + 10-15 agent-specific micro-skills
**Time**: 8-14 hours
**Complexity**: High
**Success Rate**: >97% planning accuracy, 100% recovery rate

---

### 3. End-to-End Feature Shipping

**When**: "Complete feature from research to deployment"

**Sequence**:
```yaml
1. feature-dev-complete (12-stage workflow)
   Stage 1: gemini-search (research)
   Stage 2: research-driven-planning
   Stage 3: sparc-methodology (architecture)
   Stage 4: parallel-swarm-implementation
   Stage 5: functionality-audit
   Stage 6: theater-detection-audit
   Stage 7: sop-code-review
   Stage 8: style-audit
   Stage 9: quick-quality-check
   Stage 10: production-readiness
   Stage 11: cicd-intelligent-recovery
   Stage 12: deployment-readiness
```

**Skills Used**: 12 skills
**Time**: 1-2 days
**Complexity**: Very High

---

### 4. Bug Fix with Root Cause Analysis

**When**: "Production bug", "Critical failure", "Mysterious error"

**Sequence**:
```yaml
1. smart-bug-fix
   - Intelligent debugging
   - Automated fixes
   - Regression prevention

2. sop-dogfooding-quality-detection (optional)
   - Run connascence analysis
   - Detect code quality violations
   - Store findings in Memory MCP

3. sop-dogfooding-pattern-retrieval (optional)
   - Search Memory MCP for similar bugs
   - Apply proven fixes

4. functionality-audit
   - Sandbox testing
   - Verify fix works

5. cicd-intelligent-recovery
   - Automated test suite
   - Ensure no regressions
```

**Skills Used**: 5 skills
**Time**: 1-3 hours
**Complexity**: Medium

---

### 5. Rapid Prototyping

**When**: "Quick proof of concept", "Validate idea fast"

**Sequence**:
```yaml
1. cascade-orchestrator
   Stage 1: gemini-search (research)
   Stage 2: codex-auto (rapid prototyping in sandbox)
   Stage 3: functionality-audit (does it work?)
   Stage 4: quick-quality-check (fast validation)
```

**Skills Used**: 4 skills
**Time**: 30-60 minutes
**Complexity**: Low

---

## Operations Playbooks

### 6. Production Deployment

**When**: "Deploy to production", "Release new version"

**Sequence**:
```yaml
1. production-readiness
   - Complete audit pipeline
   - Security validation
   - Performance benchmarks

2. deployment-readiness
   - Infrastructure requirements
   - Monitoring plan
   - Incident response plan
   - Rollback strategy

3. docker-containerization
   - Multi-stage builds
   - Security scanning (Trivy)
   - Optimization

4. cicd-intelligent-recovery
   - Automated deployment
   - Health checks
   - Rollback if needed
```

**Skills Used**: 4 skills
**Time**: 2-4 hours
**Complexity**: High

---

### 7. CI/CD Pipeline Setup

**When**: "Setup GitHub Actions", "Automate testing/deployment"

**Sequence**:
```yaml
1. github-workflow-automation
   - Intelligent CI/CD pipelines
   - Self-organizing workflows
   - Automated optimization

2. cicd-intelligent-recovery
   - Failure recovery
   - Root cause analysis
   - Auto-fix patterns

3. hooks-automation
   - Lifecycle event hooks
   - Pre-task validation
   - Post-task cleanup
```

**Skills Used**: 3 skills
**Time**: 3-5 hours
**Complexity**: Medium

---

### 8. Infrastructure Scaling

**When**: "Scale to handle 10x traffic", "Multi-region deployment"

**Sequence**:
```yaml
1. cloud-platforms
   - AWS/GCP/Azure selection
   - Serverless vs containers

2. kubernetes-specialist
   - K8s orchestration
   - Helm charts
   - Service mesh (Istio)

3. terraform-iac
   - Infrastructure as code
   - Multi-cloud provisioning
   - State management

4. opentelemetry-observability
   - Distributed tracing
   - Metrics collection
   - APM integration
```

**Skills Used**: 4 skills
**Time**: 1-2 weeks
**Complexity**: Very High

---

### 9. Performance Optimization

**When**: "App is slow", "Reduce latency", "Optimize database"

**Sequence**:
```yaml
1. performance-analysis
   - Comprehensive analysis
   - Bottleneck detection
   - Profiling

2. cascade-orchestrator
   Stage 1: sql-database-specialist (if DB bottleneck)
   Stage 2: frontend-performance-optimizer (if UI bottleneck)
   Stage 3: docker-containerization (if container overhead)
```

**Skills Used**: 2-4 skills
**Time**: 4-8 hours
**Complexity**: Medium-High

---

## Research Playbooks

### 10. Deep Research SOP (Academic ML) 🔥 FLAGSHIP

**When**: "NeurIPS submission", "Systematic ML research", "Reproducible experiments"

**Sequence**:
```yaml
# PHASE 1: FOUNDATIONS (2-4 weeks) → Quality Gate 1
1. deep-research-orchestrator (Phase 1)
   Pipeline A: literature-synthesis
   Pipeline B: data-steward → /init-datasheet, bias-audit
   Pipeline C: researcher → /prisma-init (if systematic review)
   Pipeline D: baseline-replication (±1% tolerance)

2. gate-validation --gate 1 (GO/NO-GO)
   Requirements:
   - Literature review ≥50 papers
   - Datasheet ≥90% complete
   - Ethics review APPROVED
   - Baseline within ±1%
   - Reproducibility package tested

# PHASE 2: DEVELOPMENT (6-12 weeks) → Quality Gate 2
3. deep-research-orchestrator (Phase 2)
   Pipeline D: method-development
   Pipeline E: holistic-evaluation (6 dimensions)
   Pipeline F: ethics-agent → /assess-risks --gate 2

4. gate-validation --gate 2 (GO/NO-GO)
   Requirements:
   - Novel method outperforms baseline
   - Ablation studies ≥5 components
   - Holistic eval 6+ dimensions
   - Ethics APPROVED
   - Method card ≥90% complete

# PHASE 3: PRODUCTION (2-4 weeks) → Quality Gate 3
5. deep-research-orchestrator (Phase 3)
   Pipeline G: reproducibility-audit
   Pipeline H: deployment-readiness
   Pipeline I: research-publication

6. gate-validation --gate 3 (GO/NO-GO → DEPLOY)
   Requirements:
   - Model card ≥90% complete
   - Reproducibility tested 3/3 runs
   - DOIs assigned
   - Code public (GitHub)
   - Ethics APPROVED
   - Deployment plan validated
```

**Skills Used**: 10 skills (baseline-replication, method-development, holistic-evaluation, literature-synthesis, reproducibility-audit, deployment-readiness, research-publication, gate-validation)
**Agents Used**: 9 agents (data-steward, ethics-agent, archivist, evaluator, researcher, system-architect, coder, tester, reviewer)
**Time**: 2-6 months
**Complexity**: Maximum

---

### 11. Quick Investigation

**When**: "Research how X works", "Understand technology Y"

**Sequence**:
```yaml
1. researcher
   - Quick literature review
   - Gather documentation
   - Identify key concepts

2. gemini-search (if web research needed)
   - Grounded search
   - Latest information
   - Multiple sources
```

**Skills Used**: 2 skills
**Time**: 30-60 minutes
**Complexity**: Low

---

### 12. Planning & Architecture

**When**: "Design new system", "Architecture review"

**Sequence**:
```yaml
1. intent-analyzer
   - Deep requirements analysis
   - Constraint detection
   - Socratic clarification

2. interactive-planner
   - Multi-select questions
   - Gather requirements
   - Clarify constraints

3. research-driven-planning
   - Technology selection
   - Risk mitigation
   - 5x pre-mortem
   - Multi-agent consensus
```

**Skills Used**: 3 skills
**Time**: 2-4 hours
**Complexity**: Medium

---

### 13. Literature Review

**When**: "Systematic review needed", "SOTA analysis", "Research gap identification"

**Sequence**:
```yaml
1. literature-synthesis
   - Multi-database search
   - PRISMA 2020 compliant
   - Citation management
   - Gap analysis

2. researcher → /prisma-init (if systematic)
   - Protocol document
   - Search strategy
   - Inclusion/exclusion criteria
```

**Skills Used**: 2 skills
**Time**: 1-2 weeks
**Complexity**: Medium

---

## Security Playbooks

### 14. Security Audit

**When**: "Pre-production security check", "Compliance requirement"

**Sequence**:
```yaml
1. network-security-setup
   - Sandbox isolation
   - Network boundaries
   - Access controls

2. compliance
   - WCAG accessibility
   - Regulatory requirements
   - Standards compliance

3. reverse-engineering-quick-triage (if binary analysis needed)
   - String reconnaissance
   - Static analysis
   - IOC extraction

4. sop-code-review
   - Security-focused review
   - Vulnerability detection
   - Best practices
```

**Skills Used**: 4 skills
**Time**: 4-8 hours
**Complexity**: High

---

### 15. Compliance Validation

**When**: "SOC2 audit", "GDPR compliance", "HIPAA requirements"

**Sequence**:
```yaml
1. compliance
   - Requirements mapping
   - Gap analysis
   - Remediation plan

2. wcag-accessibility
   - WCAG 2.1 AA/AAA
   - Screen reader testing
   - Keyboard navigation

3. network-security-setup
   - Network isolation
   - Access policies
   - Audit logging
```

**Skills Used**: 3 skills
**Time**: 1-2 weeks
**Complexity**: High

---

### 16. Reverse Engineering (Advanced)

**When**: "Malware analysis", "Binary vulnerability research", "CTF challenges"

**Sequence**:
```yaml
1. sandbox-configurator
   - Isolated environment
   - Network disabled
   - VM/Docker setup

2. reverse-engineering-quick-triage
   - Level 1: String reconnaissance (≤30 min)
   - Level 2: Static analysis with Ghidra (1-2 hrs)

3. reverse-engineering-deep-analysis (if needed)
   - Level 3: Dynamic analysis with GDB (2-4 hrs)
   - Level 4: Symbolic execution with Angr (2-4 hrs)

4. reverse-engineering-firmware-analysis (if IoT/embedded)
   - Level 5: Firmware extraction (2-8 hrs)
   - binwalk + QEMU + firmadyne
```

**Skills Used**: 4 skills
**Time**: 2-16 hours (depending on depth)
**Complexity**: Very High
**Security**: ⚠️ VM/Docker/E2B sandboxing REQUIRED

---

## Quality Playbooks

### 17. Quick Quality Check

**When**: "Fast validation before commit"

**Sequence**:
```yaml
1. quick-quality-check
   - Parallel lint/security/tests
   - Instant feedback (5-10 seconds)
   - Theater detection
   - Security scan (OWASP Top 10)
   - Basic tests
```

**Skills Used**: 1 skill
**Time**: 5-30 seconds
**Complexity**: Low

---

### 18. Comprehensive Code Review

**When**: "PR review", "Pre-merge validation"

**Sequence**:
```yaml
1. code-review-assistant
   - Multi-agent swarm review
   - 5 specialist reviewers:
     * Security: Vulnerability scan
     * Performance: Bottleneck detection
     * Style: Code quality audit
     * Tests: Coverage analysis
     * Docs: Documentation completeness

2. theater-detection-audit
   - 6-agent Byzantine consensus
   - Sandbox execution testing
   - Real implementation verification

3. functionality-audit
   - Systematic debugging
   - Sandbox testing
   - Best practices validation
```

**Skills Used**: 3 skills
**Time**: 1-2 hours
**Complexity**: Medium

---

### 19. Dogfooding Cycle (Self-Improvement)

**When**: "Improve code quality continuously"

**Sequence**:
```yaml
1. sop-dogfooding-quality-detection
   - Run Connascence analysis
   - Detect violations (God Objects, Parameter Bombs, etc.)
   - Store in Memory MCP with WHO/WHEN/PROJECT/WHY

2. sop-dogfooding-pattern-retrieval
   - Vector search Memory MCP for similar violations
   - Rank proven fixes
   - Optionally apply fixes

3. sop-dogfooding-continuous-improvement
   - Full cycle orchestration
   - Sandbox testing
   - Metrics tracking
   - Continuous learning
```

**Skills Used**: 3 skills
**Time**: 60-120 seconds
**Complexity**: Medium

---

## Platform Playbooks

### 20. Machine Learning Pipeline

**When**: "Train ML model", "ML ops deployment"

**Sequence**:
```yaml
1. machine-learning
   - Data pipeline
   - Model training
   - Evaluation
   - Deployment

2. agentdb-learning-plugins (if RL needed)
   - 9 RL algorithms
   - Decision Transformer
   - Q-Learning, SARSA, Actor-Critic

3. holistic-evaluation
   - Multi-metric evaluation
   - Fairness analysis
   - Robustness testing
```

**Skills Used**: 3 skills
**Time**: Hours to days
**Complexity**: Very High

---

### 21. Vector Search & RAG System

**When**: "Build semantic search", "RAG implementation"

**Sequence**:
```yaml
1. agentdb-vector-search
   - Semantic search
   - Document retrieval
   - HNSW indexing

2. agentdb-performance-optimization
   - Quantization (4-32x memory reduction)
   - 150x faster search
   - Batch operations

3. agentdb-memory-patterns
   - Session memory
   - Long-term storage
   - Pattern learning
```

**Skills Used**: 3 skills
**Time**: 4-8 hours
**Complexity**: Medium-High

---

### 22. Distributed Neural Training

**When**: "Train large models", "Distributed compute"

**Sequence**:
```yaml
1. flow-nexus-neural
   - E2B sandbox clusters
   - Distributed training
   - Multi-node coordination

2. reasoningbank-with-agentdb
   - Pattern learning
   - 46% faster performance
   - 88% success rate
```

**Skills Used**: 2 skills
**Time**: Hours to days
**Complexity**: Very High

---

## GitHub Playbooks

### 23. Pull Request Management

**When**: "Review PR", "Merge workflow"

**Sequence**:
```yaml
1. github-code-review
   - AI swarm PR review
   - Multi-agent coordination
   - Automated feedback

2. code-review-assistant
   - 5 specialist reviewers
   - Comprehensive analysis
   - Auto-fix suggestions
```

**Skills Used**: 2 skills
**Time**: 30-60 minutes
**Complexity**: Medium

---

### 24. Release Management

**When**: "Cut release", "Version deployment"

**Sequence**:
```yaml
1. github-release-management
   - Automated versioning
   - Changelog generation
   - Deployment coordination
   - Rollback management

2. deployment-readiness
   - Production validation
   - Health checks
   - Monitoring setup
```

**Skills Used**: 2 skills
**Time**: 1-2 hours
**Complexity**: Medium

---

### 25. Multi-Repo Coordination

**When**: "Sync multiple repos", "Monorepo management"

**Sequence**:
```yaml
1. github-multi-repo
   - Cross-repo coordination
   - Version alignment
   - Dependency sync

2. github-project-management
   - Issue tracking
   - Project board automation
   - Sprint planning
```

**Skills Used**: 2 skills
**Time**: 2-4 hours
**Complexity**: High

---

## Specialist Playbooks

### 26. Frontend Development

**When**: "Build React app", "UI component library"

**Sequence**:
```yaml
1. react-specialist
   - Modern React 18
   - Hooks, Composition API
   - State management (Zustand)
   - Performance optimization

2. frontend-specialists (parent skill)
   - CSS optimization
   - Accessibility (WCAG)
   - Performance (Core Web Vitals)
```

**Skills Used**: 2 skills
**Time**: 4-8 hours
**Complexity**: Medium

---

### 27. Backend Development

**When**: "Build API", "Backend services"

**Sequence**:
```yaml
1. language-specialists
   - python-specialist: FastAPI, Django, Flask
   - typescript-specialist: Nest.js, Express

2. sql-database-specialist
   - Query optimization
   - Schema design
   - Indexing strategies

3. sop-api-development
   - Systematic API workflow
   - REST/GraphQL best practices
```

**Skills Used**: 3 skills
**Time**: 6-12 hours
**Complexity**: Medium-High

---

### 28. Full-Stack with Docker

**When**: "Complete application stack", "Containerized deployment"

**Sequence**:
```yaml
1. language-specialists
   - Frontend: react-specialist or vue-specialist
   - Backend: python-specialist or typescript-specialist

2. docker-containerization
   - Multi-stage builds
   - Security scanning (Trivy)
   - Docker Compose orchestration

3. terraform-iac
   - Infrastructure provisioning
   - Multi-cloud deployment
```

**Skills Used**: 3-4 skills
**Time**: 1-2 days
**Complexity**: High

---

### 29. Infrastructure as Code

**When**: "Provision infrastructure", "Multi-cloud setup"

**Sequence**:
```yaml
1. terraform-iac
   - Multi-cloud provisioning
   - State management
   - Modules and GitOps

2. cloud-platforms
   - aws-specialist: AWS CDK, CloudFormation
   - kubernetes-specialist: K8s, Helm, operators

3. opentelemetry-observability
   - Distributed tracing
   - Metrics and logging
   - APM integration
```

**Skills Used**: 3 skills
**Time**: 1-2 weeks
**Complexity**: Very High

---

## How to Use This System

### Auto-Triggering Pattern

**You DON'T manually select skills. Just describe what you want:**

```bash
# ❌ Old way: Manual skill selection
"Use parallel-swarm-implementation skill to build REST API"

# ✅ New way: Natural language → automatic routing
"Build a REST API for user management"

# What happens automatically:
# 1. intent-analyzer detects feature implementation intent
# 2. cascade-orchestrator selects appropriate playbook
# 3. Skills execute in optimal sequence
# 4. Result: Production-ready API
```

### Playbook Selection Logic

**System auto-selects playbook based on intent:**

| User Intent | Detected Keywords | Playbook Selected |
|-------------|-------------------|-------------------|
| Feature implementation | "build", "add", "implement" | Simple Feature OR Three-Loop (complexity-dependent) |
| Bug fix | "bug", "error", "failing" | Bug Fix with RCA |
| Research | "research", "investigate", "understand" | Quick Investigation OR Deep Research (scope-dependent) |
| Deployment | "deploy", "release", "production" | Production Deployment |
| Security | "security", "audit", "compliance" | Security Audit |
| Code review | "review", "PR", "merge" | Comprehensive Code Review |
| Performance | "slow", "optimize", "performance" | Performance Optimization |

---

## Key Insights

1. **122 skills organized into 10 categories** (Delivery, Operations, Research, Security, Quality, Platform, GitHub, Foundry, Workflow, Utilities)

2. **25+ comprehensive playbooks** covering entire software development lifecycle

3. **Auto-routing eliminates decision paralysis** - just describe what you want

4. **Two flagship playbooks**:
   - **Three-Loop**: >97% planning accuracy, 100% recovery rate
   - **Deep Research SOP**: 2-6 month research lifecycle with 3 quality gates

5. **Existing orchestration capabilities**:
   - `intent-analyzer`: Deep intent analysis
   - `cascade-orchestrator`: Multi-skill workflow coordination
   - `deep-research-orchestrator`: Academic research lifecycle
   - `parallel-swarm-implementation`: 6-10 agent parallel execution

6. **No need for separate "orchestration-router"** - routing is distributed across intent-analyzer + cascade-orchestrator + deep-research-orchestrator

---

**Next Step**: Update SKILL-PLAYBOOK.md to reflect these actual skill-based playbooks!