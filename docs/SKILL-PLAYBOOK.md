# Skill Playbook System
## Orchestrated Skill Sequences for Optimal Workflows

**Version**: 1.0.0
**Updated**: 2025-11-14

---

## Overview

The **Skill Playbook** system defines optimized sequences of skills for different types of tasks. Instead of manually selecting skills, the system uses:

1. **intent-analyzer** - Analyzes user request and determines intent
2. **prompt-architect** - Improves the request for clarity
3. **orchestration-router** - Selects the right playbook
4. **Skill Sequence** - Executes coordinated skill chain

This creates a **zero-decision paralysis** workflow where every request follows a proven path.

---

## Universal Workflow (All Requests)

```
User Request
    ↓
📋 Phase 0: Intent Analysis & Prompt Optimization
    ↓
🔍 intent-analyzer
    ├─ Analyzes explicit and implicit goals
    ├─ Detects constraints and context
    ├─ Maps to probabilistic intent (>80% confidence)
    └─ Socratic clarification if needed (<80%)
    ↓
✨ prompt-architect
    ├─ Evidence-based prompt optimization
    ├─ Clarity enhancement
    ├─ Constraint specification
    └─ Success criteria definition
    ↓
🎯 orchestration-router
    ├─ Keyword extraction (agent count, complexity, patterns)
    ├─ Decision tree routing (Priority 1-4 logic)
    ├─ Playbook selection with rationale
    └─ Automatic playbook execution
    ↓
⚡ Selected Playbook Executes
    ├─ Coordination skill (swarm/hierarchical)
    ├─ Domain-specific skills in sequence
    ├─ Quality validation skills
    └─ Completion and handoff
```

---

## Playbook Categories

### 🚀 Delivery Playbooks (Feature Development)
**Trigger**: "build", "implement", "create feature", "develop"

### ⚙️ Operations Playbooks (Infrastructure & Deployment)
**Trigger**: "deploy", "CI/CD", "infrastructure", "release"

### 🔬 Research Playbooks (Discovery & Analysis)
**Trigger**: "research", "analyze", "investigate", "study"

### 🔒 Security Playbooks (Hardening & Compliance)
**Trigger**: "security", "audit", "compliance", "vulnerability"

### 🎨 Specialist Playbooks (Domain-Specific)
**Trigger**: Technology stack keywords (React, Python, ML, etc.)

---

## Detailed Playbooks

### Delivery Playbook 1: Simple Feature Implementation

**When**: Single feature, clear requirements, low complexity

**Sequence**:
```yaml
1. intent-analyzer → Clarify feature requirements
2. prompt-architect → Enhance feature specification
3. orchestration-router → Route to simple-feature playbook
4. sparc-methodology → 5-phase structured development
5. functionality-audit → Validate implementation works
6. code-review-assistant → Quality check
7. quick-quality-check → Fast validation (lint/security/tests)
```

**Example Triggers**:
- "Add user authentication"
- "Implement dark mode toggle"
- "Create contact form"

---

### Delivery Playbook 2: Complex Feature (Three-Loop)

**When**: Complex feature, multiple components, high stakes

**Sequence**:
```yaml
1. intent-analyzer → Deep requirements analysis
2. prompt-architect → Comprehensive spec optimization
3. orchestration-router → Route to three-loop playbook
4. research-driven-planning → Loop 1 (5x pre-mortem, risk mitigation)
5. parallel-swarm-implementation → Loop 2 (6-agent parallel execution)
6. theater-detection-audit → Byzantine consensus validation
7. cicd-intelligent-recovery → Loop 3 (automated testing + fixes)
8. production-readiness → Final deployment validation
```

**Example Triggers**:
- "Build payment processing with Stripe"
- "Implement real-time collaboration"
- "Create admin dashboard with analytics"

---

### Delivery Playbook 3: End-to-End Feature Shipping

**When**: Complete feature from research to deployment

**Sequence**:
```yaml
1. intent-analyzer → Understand full scope
2. prompt-architect → End-to-end requirements
3. orchestration-router → Route to e2e playbook
4. feature-dev-complete → 12-stage workflow
   ├─ Gemini search (best practices)
   ├─ Architecture design
   ├─ Codex prototyping
   ├─ Implementation
   ├─ Comprehensive testing
   ├─ Documentation
   └─ Deployment
5. production-validator → Deployment readiness check
```

**Example Triggers**:
- "Ship user onboarding flow"
- "Launch notification system"
- "Deploy API v2 with migration"

---

### Operations Playbook 1: Production Deployment

**When**: Deploying to production environment

**Sequence**:
```yaml
1. intent-analyzer → Deployment scope analysis
2. prompt-architect → Deployment checklist optimization
3. orchestration-router → Route to deployment playbook
4. production-readiness → Comprehensive audit
   ├─ Performance benchmarks
   ├─ Security scanning
   ├─ Monitoring setup
   ├─ Rollback plan
5. github-release-management → Automated versioning
6. cicd-intelligent-recovery → Deployment pipeline
7. sop-product-launch → Complete launch workflow
```

**Example Triggers**:
- "Deploy to production"
- "Launch new version"
- "Release v2.0"

---

### Operations Playbook 2: CI/CD Setup

**When**: Setting up or fixing CI/CD pipelines

**Sequence**:
```yaml
1. intent-analyzer → CI/CD requirements
2. prompt-architect → Pipeline optimization
3. orchestration-router → Route to cicd playbook
4. github-workflow-automation → Intelligent CI/CD pipelines
5. cicd-intelligent-recovery → Failure recovery setup
6. docker-containerization → Container optimization
7. terraform-iac → Infrastructure as code
```

**Example Triggers**:
- "Setup GitHub Actions"
- "Fix failing CI pipeline"
- "Automate deployment"

---

### Operations Playbook 3: Infrastructure Scaling

**When**: Scaling infrastructure, cloud migration

**Sequence**:
```yaml
1. intent-analyzer → Scaling requirements
2. prompt-architect → Infrastructure spec
3. orchestration-router → Route to infrastructure playbook
4. aws-specialist / kubernetes-specialist → Platform setup
5. terraform-iac → Infrastructure provisioning
6. opentelemetry-observability → Monitoring setup
7. performance-analysis → Bottleneck detection
```

**Example Triggers**:
- "Scale to 10M users"
- "Migrate to Kubernetes"
- "Setup autoscaling"

---

### Research Playbook 1: Quick Investigation

**When**: Fast research, simple question

**Sequence**:
```yaml
1. intent-analyzer → Research question clarification
2. prompt-architect → Query optimization
3. orchestration-router → Route to quick-research playbook
4. researcher → Gemini search + pattern recognition
5. intent-analyzer → Synthesize findings
```

**Example Triggers**:
- "What's the best approach for...?"
- "Research authentication methods"
- "Compare React vs Vue"

---

### Research Playbook 2: Comprehensive Research (Deep Research SOP)

**When**: Academic research, systematic analysis

**Sequence**:
```yaml
1. intent-analyzer → Research objectives
2. prompt-architect → Research question refinement
3. orchestration-router → Route to deep-research playbook
4. literature-synthesis → PRISMA 2020 systematic review
5. baseline-replication → Reproduce existing results
6. method-development → Novel approach development
7. holistic-evaluation → Comprehensive evaluation
8. gate-validation → Quality Gate approval
9. reproducibility-audit → ACM artifact evaluation
10. research-publication → Paper writing + submission
```

**Example Triggers**:
- "Conduct systematic literature review"
- "Replicate baseline results"
- "Publish research paper"

---

### Research Playbook 3: Planning & Architecture

**When**: New project planning, architecture design

**Sequence**:
```yaml
1. intent-analyzer → Project scope analysis
2. prompt-architect → Requirements specification
3. orchestration-router → Route to planning playbook
4. research-driven-planning → Research + 5x pre-mortem
5. sparc-methodology → Architecture phase
6. api-designer → API contract design (if applicable)
7. interactive-planner → Gather additional requirements
```

**Example Triggers**:
- "Plan new microservices architecture"
- "Design REST API for e-commerce"
- "Architecture for mobile app"

---

### Security Playbook 1: Security Audit

**When**: Security review, vulnerability assessment

**Sequence**:
```yaml
1. intent-analyzer → Security scope
2. prompt-architect → Audit criteria
3. orchestration-router → Route to security-audit playbook
4. network-security-setup → Network isolation
5. code-review-assistant → Security-focused review
6. production-readiness → Security checklist
7. sop-code-review → Systematic security review
```

**Example Triggers**:
- "Security audit before launch"
- "Find vulnerabilities"
- "OWASP Top 10 compliance"

---

### Security Playbook 2: Compliance Validation

**When**: Regulatory compliance, accessibility

**Sequence**:
```yaml
1. intent-analyzer → Compliance requirements
2. prompt-architect → Standards specification
3. orchestration-router → Route to compliance playbook
4. wcag-accessibility → WCAG 2.1 AA/AAA compliance
5. verification-quality → Quality assurance
6. reproducibility-audit → Audit trail creation
```

**Example Triggers**:
- "WCAG accessibility compliance"
- "SOC 2 compliance check"
- "GDPR validation"

---

### Security Playbook 3: Reverse Engineering

**When**: Malware analysis, binary analysis, security research

**Sequence**:
```yaml
1. intent-analyzer → Analysis objectives
2. prompt-architect → Analysis parameters
3. orchestration-router → Route to reverse-eng playbook
4. sandbox-configurator → Isolated environment setup
5. reverse-engineering-quick/deep/firmware → Level-appropriate analysis
6. network-security-setup → Network isolation
7. verification-quality → Results validation
```

**Example Triggers**:
- "Analyze malware sample"
- "Reverse engineer binary"
- "Firmware vulnerability assessment"

---

### Specialist Playbook 1: Frontend Development

**When**: React, Vue, UI/UX work

**Sequence**:
```yaml
1. intent-analyzer → Frontend requirements
2. prompt-architect → Component specification
3. orchestration-router → Route to frontend playbook
4. react-specialist / vue-developer → Component implementation
5. ui-component-builder → Reusable component library
6. accessibility-specialist → WCAG compliance
7. frontend-performance-optimizer → Core Web Vitals
8. visual-regression-agent → UI consistency testing
```

**Example Triggers**:
- "Build React dashboard"
- "Create Vue component library"
- "Optimize frontend performance"

---

### Specialist Playbook 2: Backend Development

**When**: API development, server-side logic

**Sequence**:
```yaml
1. intent-analyzer → API requirements
2. prompt-architect → API specification
3. orchestration-router → Route to backend playbook
4. api-designer → API contract design
5. python-specialist / typescript-specialist → Implementation
6. database-design-specialist → Schema design
7. cache-strategy-agent → Caching optimization
8. performance-testing-agent → Load testing
```

**Example Triggers**:
- "Build REST API"
- "Create GraphQL backend"
- "Optimize database queries"

---

### Specialist Playbook 3: Machine Learning

**When**: ML model development, training, deployment

**Sequence**:
```yaml
1. intent-analyzer → ML objectives
2. prompt-architect → Model requirements
3. orchestration-router → Route to ml playbook
4. machine-learning → Complete ML workflow
5. data-steward → Dataset documentation + bias audit
6. ml-expert → Advanced model development
7. holistic-evaluation → Multi-metric evaluation
8. deployment-readiness → Production ML deployment
```

**Example Triggers**:
- "Train neural network"
- "Build recommendation system"
- "Deploy ML model to production"

---

## Priority Routing Logic (orchestration-router)

**The orchestration-router skill uses this decision tree:**

### Priority 1: Explicit Workflow Signals

| Signal | Playbook |
|--------|----------|
| "three-loop", "research-driven", "pre-mortem" | Delivery Playbook 2 (Three-Loop) |
| "end-to-end", "ship feature", "complete workflow" | Delivery Playbook 3 (E2E) |
| "deep research", "systematic review", "publish paper" | Research Playbook 2 (Deep SOP) |
| "deploy to production", "release", "launch" | Operations Playbook 1 (Deployment) |

### Priority 2: Complexity Signals

| Agent Count / Complexity | Playbook |
|--------------------------|----------|
| "6+ agents", "complex", "high stakes" | Three-Loop (Delivery 2) |
| "parallel swarm", "multi-agent" | Swarm Orchestration |
| "simple", "quick", "single component" | Simple Feature (Delivery 1) |

### Priority 3: Domain Signals

| Domain Keywords | Playbook |
|-----------------|----------|
| React, Vue, frontend, UI | Specialist Playbook 1 (Frontend) |
| API, backend, database | Specialist Playbook 2 (Backend) |
| ML, neural network, training | Specialist Playbook 3 (ML) |
| CI/CD, deployment, infrastructure | Operations Playbooks |
| security, audit, compliance | Security Playbooks |

### Priority 4: Default Fallback

If no clear signals → **Simple Feature (Delivery 1)** with SPARC methodology

---

## Playbook Execution Flow

```javascript
// Example: User says "Build payment processing with Stripe"

1. intent-analyzer
   → Intent: Complex feature implementation
   → Confidence: 95%
   → Constraints: Payment security, PCI compliance
   → Proceed to prompt optimization

2. prompt-architect
   → Enhanced: "Implement Stripe payment processing with:
      - Secure checkout flow
      - Webhook handling for payment events
      - PCI DSS compliance
      - Test mode + production mode
      - Error handling and retry logic
      - Payment method management"

3. orchestration-router
   → Detected signals: "complex", "payment", "security"
   → Selected: Delivery Playbook 2 (Three-Loop)
   → Rationale: High complexity + security requirements

4. Playbook Execution:
   Loop 1 (research-driven-planning)
     → Research Stripe best practices
     → 5x pre-mortem (payment failures, security risks, etc.)
     → Risk mitigation strategies

   Loop 2 (parallel-swarm-implementation)
     → 6 agents in parallel:
        - researcher: Stripe API patterns
        - coder: Payment endpoints
        - reviewer: Security audit
        - tester: Payment testing
        - documenter: API docs
        - theater-detector: Reality validation

   Loop 3 (cicd-intelligent-recovery)
     → Automated testing
     → Failure detection + fixes
     → Production validation

5. Result: Production-ready Stripe integration in 2-4 hours
```

---

## Adding New Playbooks

To add a custom playbook:

1. **Define the sequence** in this document
2. **Add trigger patterns** to orchestration-router skill
3. **Test the playbook** with sample requests
4. **Document in** this file

Example template:

```yaml
### [Category] Playbook N: [Name]

**When**: [Trigger conditions]

**Sequence**:
1. intent-analyzer → [Purpose]
2. prompt-architect → [Enhancement]
3. orchestration-router → Route to [playbook-name]
4. [coordination-skill] → [Orchestration]
5. [domain-skill-1] → [Specific task]
6. [domain-skill-2] → [Specific task]
7. [validation-skill] → [Quality check]

**Example Triggers**:
- "[Example 1]"
- "[Example 2]"
- "[Example 3]"
```

---

## Playbook Benefits

### 1. Zero Decision Paralysis
Users don't need to know which skills to use - the system auto-routes.

### 2. Proven Patterns
Each playbook represents a battle-tested workflow that works.

### 3. Consistent Quality
Every request follows a structured path with validation steps.

### 4. Adaptive Routing
The system learns from user corrections and adjusts routing.

### 5. Transparent Reasoning
Every playbook selection includes rationale and alternatives.

---

## Related Documentation

- [Orchestration Router Skill](../skills/orchestration/orchestration-router/SKILL.md)
- [Intent Analyzer Skill](../skills/research/intent-analyzer/SKILL.md)
- [Prompt Architect Skill](../skills/foundry/prompt-architect/SKILL.md)
- [Skill Directory](../skills/README.md)
- [Plugin Documentation](PLUGIN-VALIDATION-REPORT.md)

---

**Last Updated**: 2025-11-14
**Version**: 1.0.0
**Maintained By**: ruv-sparc-three-loop-system team
