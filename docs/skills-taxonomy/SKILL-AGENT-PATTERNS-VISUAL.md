# Skill-Agent Coordination Patterns - Visual Reference

**Purpose**: Quick visual reference for the 5 main coordination patterns discovered
**Date**: 2025-11-02
**Total Patterns**: 5
**Skills Using Patterns**: 130

---

## Pattern 1: Hierarchical Multi-Phase Workflow

**Used By**: feature-dev-complete, smart-bug-fix, sparc-methodology, sop-product-launch (15 skills total)

### Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL MULTI-PHASE WORKFLOW                    │
│                    (7-phase feature development)                        │
└─────────────────────────────────────────────────────────────────────────┘

Phase 1: Research & Requirements
┌────────────┐
│ researcher │──────► [Memory: swarm/feature-dev/requirements/*]
└────────────┘
      │
      ↓ (handoff via memory)

Phase 2: Architecture Design
┌─────────────────┐
│ system-architect│──────► [Memory: swarm/feature-dev/architecture/*]
└─────────────────┘
      │
      ↓

Phase 3: Planning
┌─────────┐
│ planner │──────► [Memory: swarm/feature-dev/planning/*]
└─────────┘
      │
      ↓

Phase 4: TDD Implementation (PARALLEL)
┌────────┐                    ┌───────┐
│ tester │◄──────►│Memory│◄──────►│ coder │
└────────┘         └──────┘        └───────┘
   │                                   │
   │ [test specs]        [implementation] │
   └─────────────┬─────────────┘
                 ↓

Phase 5: Code Review
┌──────────┐
│ reviewer │──────► [Memory: swarm/feature-dev/review-findings/*]
└──────────┘
      │
      ├─────► (if issues) ──────► BACK TO PHASE 4
      │
      ↓ (if approved)

Phase 6 & 7: Documentation + Deployment (PARALLEL)
┌──────────┐              ┌───────────────┐
│ api-docs │              │ cicd-engineer │
└──────────┘              └───────────────┘
      │                           │
      └───────────┬───────────┘
                  ↓
            [COMPLETE]

COORDINATOR: hierarchical-coordinator
TOPOLOGY: Hierarchical tree
MEMORY PATTERN: Structured namespaces per phase
QUALITY GATES: After Phase 2, 4, 5
```

### Example Assignments

| Skill | Phases | Agents |
|-------|--------|--------|
| feature-dev-complete | 7 | researcher, system-architect, planner, tester, coder, reviewer, api-docs, cicd-engineer |
| smart-bug-fix | 7 | researcher (RCA), coder (fix), tester (validation), reviewer (QA), performance-analyzer |
| sparc-methodology | 5 | sparc-coord, specification, pseudocode, architecture, refinement, sparc-coder |

---

## Pattern 2: Parallel Swarm Execution

**Used By**: parallel-swarm-implementation, quick-quality-check, code-review-assistant (12 skills total)

### Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      PARALLEL SWARM EXECUTION                           │
│                  (concurrent multi-agent processing)                    │
└─────────────────────────────────────────────────────────────────────────┘

                        ┌────────────────────┐
                        │   COORDINATOR      │
                        │ (hierarchical-     │
                        │  coordinator)      │
                        └────────┬───────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ↓                       ↓                       ↓
   ┌─────────┐            ┌─────────┐            ┌─────────┐
   │ agent 1 │            │ agent 2 │            │ agent 3 │
   │(tester) │            │(security│            │ (e2e)   │
   └────┬────┘            └────┬────┘            └────┬────┘
        │                      │                      │
        │ [run tests]          │ [security scan]      │ [e2e tests]
        │                      │                      │
   [results 1]            [results 2]            [results 3]
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ↓
                    ┌─────────────────────┐
                    │ AGGREGATED RESULTS  │
                    │ (by coordinator)    │
                    └─────────────────────┘
                               ↓
                    [Memory: swarm/parallel-results/*]

COORDINATOR: hierarchical-coordinator or mesh-coordinator
TOPOLOGY: Mesh (peer-to-peer) or Star (centralized)
MEMORY PATTERN: Shared results namespace
EXECUTION: All agents run concurrently (2.8-4.4x speedup)
```

### Example Assignments

| Skill | Coordinator | Parallel Workers |
|-------|-------------|------------------|
| quick-quality-check | tester | tester, security-testing-agent, e2e-testing-specialist, performance-testing-agent |
| code-review-assistant | reviewer | security-manager, performance-analyzer, code-analyzer |
| parallel-swarm-implementation | hierarchical-coordinator | coder, tester, reviewer (concurrent) |

---

## Pattern 3: Byzantine Consensus Verification

**Used By**: theater-detection-audit, Quality Gate approvals, critical validations (4 skills total)

### Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  BYZANTINE CONSENSUS VERIFICATION                       │
│           (6+ independent validators, 67% threshold)                    │
└─────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────┐
                         │  TASK TO VERIFY │
                         │ (e.g., code is  │
                         │  real vs theater│
                         └────────┬────────┘
                                  │
      ┌───────────────────────────┼───────────────────────────┐
      │           │           │           │           │        │
      ↓           ↓           ↓           ↓           ↓        ↓
┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
│validator1││validator2││validator3││validator4││validator5││validator6│
│(reviewer)││(tester)  ││(security)││(prod-val)││(code-ana)││(audit)   │
└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘
     │           │           │           │           │           │
  [VOTE:     [VOTE:     [VOTE:     [VOTE:     [VOTE:     [VOTE:
   REAL]      REAL]      REAL]      REAL]      FAKE]      REAL]
     │           │           │           │           │           │
     └───────────┴───────────┴───────────┴───────────┴───────────┘
                                  │
                                  ↓
                    ┌──────────────────────────┐
                    │ CONSENSUS BUILDER        │
                    │ (byzantine-coordinator)  │
                    │                          │
                    │ Votes: 5 REAL, 1 FAKE   │
                    │ Threshold: 67% (4/6)     │
                    │ Result: 83% > 67%        │
                    │ DECISION: ✅ APPROVED    │
                    └──────────┬───────────────┘
                               ↓
                    [Memory: swarm/consensus/votes]
                               ↓
                         ┌─────────┐
                         │ ACTION  │
                         │APPROVED │
                         └─────────┘

COORDINATOR: byzantine-coordinator or consensus-builder
VALIDATORS: 6+ independent agents (no coordination between them)
THRESHOLD: 67%+ agreement required (Byzantine fault-tolerant)
MEMORY PATTERN: Vote records + consensus state
PREVENTS: Single-agent manipulation, theater code
```

### Example Assignments

| Skill | Consensus Builder | Validators (6+) | Purpose |
|-------|-------------------|-----------------|---------|
| theater-detection-audit | byzantine-coordinator | reviewer, tester, security-manager, production-validator, code-analyzer, audit-pipeline-orchestrator | Verify code is real implementation |
| Quality Gate 1 | evaluator | data-steward, ethics-agent, reviewer, tester, security-manager, archivist | Validate research quality |
| production-readiness | consensus-builder | production-validator, security-testing-agent, performance-testing-agent, reviewer, tester, audit-pipeline-orchestrator | Approve deployment |

---

## Pattern 4: Research-Then-Action

**Used By**: research-driven-planning, gemini-search workflows, intent-analyzer (10 skills total)

### Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      RESEARCH-THEN-ACTION PATTERN                       │
│             (sequential research → implementation handoff)              │
└─────────────────────────────────────────────────────────────────────────┘

PHASE 1: RESEARCH
┌────────────────────────────────────────────────┐
│                  RESEARCHER                    │
│                                                │
│  Activities:                                   │
│  • Gemini grounded search for best practices  │
│  • Analyze existing solutions                 │
│  • Research constraints and requirements      │
│  • Document findings and recommendations      │
│  • Create implementation blueprint            │
└────────────────┬───────────────────────────────┘
                 │
                 │ [store findings]
                 ↓
     ┌───────────────────────────┐
     │  MEMORY: swarm/research/  │
     │  • findings               │
     │  • best-practices         │
     │  • recommendations        │
     │  • blueprint              │
     └───────────┬───────────────┘
                 │
                 │ [retrieve findings]
                 ↓
PHASE 2: ACTION
┌────────────────────────────────────────────────┐
│             SPECIALIST AGENT                   │
│           (backend-dev, coder, etc.)           │
│                                                │
│  Activities:                                   │
│  • Retrieve research findings from memory     │
│  • Implement based on blueprint               │
│  • Follow best practices from research        │
│  • Apply recommendations                      │
│  • Store implementation results               │
└────────────────────────────────────────────────┘

COORDINATOR: None (sequential with memory handoff)
TOPOLOGY: Sequential pipeline
MEMORY PATTERN: Research findings namespace bridges phases
BENEFIT: Evidence-based implementation with research foundation
```

### Example Assignments

| Skill | Researcher | Specialist | Research Focus |
|-------|-----------|-----------|----------------|
| research-driven-planning | researcher | planner | Best practices, patterns, pre-mortem analysis |
| gemini-search | researcher | analyst | Web research, grounded search, fact-finding |
| intent-analyzer | researcher | planner | User intent analysis, requirement clarification |
| when-building-backend-api | researcher | backend-dev | API design patterns, security best practices |

---

## Pattern 5: Quality Gate Pipeline

**Used By**: Deep Research SOP (data-steward, ethics-agent, archivist, evaluator) (4 skills total)

### Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUALITY GATE PIPELINE PATTERN                        │
│          (3 sequential gates with GO/NO-GO decisions)                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  QUALITY GATE 1: Data & Methods                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  Validators:                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │
│  │ data-steward │    │ ethics-agent │    │   reviewer   │            │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘            │
│         │ [datasheet]        │ [ethics]          │ [quality]          │
│         └────────────────────┼───────────────────┘                     │
│                              ↓                                          │
│                      ┌───────────────┐                                 │
│                      │   EVALUATOR   │                                 │
│                      │  (gate auth)  │                                 │
│                      └───────┬───────┘                                 │
│                              │                                          │
│                    ┌─────────┴─────────┐                              │
│                    │                   │                               │
│               ❌ NO-GO            ✅ GO                                 │
│            [stop here]       [proceed to Gate 2]                       │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  QUALITY GATE 2: Model & Evaluation                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Validators:                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │
│  │ ethics-agent │    │  archivist   │    │    tester    │            │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘            │
│         │ [fairness]         │ [repro]           │ [validation]       │
│         └────────────────────┼───────────────────┘                     │
│                              ↓                                          │
│                      ┌───────────────┐                                 │
│                      │   EVALUATOR   │                                 │
│                      │  (gate auth)  │                                 │
│                      └───────┬───────┘                                 │
│                              │                                          │
│                    ┌─────────┴─────────┐                              │
│                    │                   │                               │
│               ❌ NO-GO            ✅ GO                                 │
│            [stop here]       [proceed to Gate 3]                       │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  QUALITY GATE 3: Production & Artifacts                                │
├─────────────────────────────────────────────────────────────────────────┤
│  Validators:                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │
│  │  archivist   │    │ ethics-agent │    │ prod-validator│            │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘            │
│         │ [DOI/card]         │ [final]           │ [prod-ready]       │
│         └────────────────────┼───────────────────┘                     │
│                              ↓                                          │
│                      ┌───────────────┐                                 │
│                      │   EVALUATOR   │                                 │
│                      │  (gate auth)  │                                 │
│                      └───────┬───────┘                                 │
│                              │                                          │
│                    ┌─────────┴─────────┐                              │
│                    │                   │                               │
│               ❌ NO-GO            ✅ GO                                 │
│            [stop here]       [PRODUCTION DEPLOYMENT]                   │
└─────────────────────────────────────────────────────────────────────────┘

COORDINATOR: evaluator (final authority for all gates)
VALIDATORS: 2-3 domain specialists per gate
TOPOLOGY: Sequential gates with approval workflow
MEMORY PATTERN: `quality-gate-{1,2,3}/*` + gate approval flags
PURPOSE: Rigorous validation for research/production systems
```

### Example Assignments

| Skill | Gate Authority | Gate 1 Validators | Gate 2 Validators | Gate 3 Validators |
|-------|----------------|-------------------|-------------------|-------------------|
| data-steward | evaluator | data-steward, ethics-agent | ethics-agent, archivist | archivist, production-validator |
| Deep Research SOP | evaluator | data-steward, ethics-agent, reviewer | ethics-agent, archivist, tester | archivist, ethics-agent, production-validator |

---

## Pattern Selection Guide

### When to Use Each Pattern

**Use Hierarchical Multi-Phase** when:
- Task requires 5+ distinct phases
- Each phase builds on previous (sequential dependencies)
- Need quality gates between phases
- Complex feature development or systematic workflows
- Examples: feature-dev-complete, smart-bug-fix, sparc-methodology

**Use Parallel Swarm Execution** when:
- Need fast execution (time-critical)
- Tasks can run independently
- Results aggregated at end
- Multiple validation types needed
- Examples: quick-quality-check, code-review-assistant, parallel-swarm-implementation

**Use Byzantine Consensus** when:
- High-stakes decision (production deployment, security approval)
- Need Byzantine fault tolerance
- Prevent single-agent manipulation
- Require verifiable consensus
- Examples: theater-detection-audit, production-readiness, Quality Gate approvals

**Use Research-Then-Action** when:
- Need evidence-based implementation
- Unknown problem domain (research first)
- Best practices research required
- Blueprint needed before implementation
- Examples: research-driven-planning, gemini-search, intent-analyzer

**Use Quality Gate Pipeline** when:
- Research or production systems
- Multiple validation domains (ethics, data quality, reproducibility)
- Regulatory compliance needed
- Rigorous validation required
- Examples: Deep Research SOP, clinical trials, high-assurance systems

---

## Pattern Combinations

Many complex skills combine multiple patterns:

### Example: feature-dev-complete

**Primary Pattern**: Hierarchical Multi-Phase (7 phases)
**Phase 4 Uses**: Parallel Execution (tester + coder concurrent)
**Phase 5 May Use**: Byzantine Consensus (if critical feature)
**Phase 1 Uses**: Research-Then-Action (research → architecture)

### Example: production-readiness

**Primary Pattern**: Quality Gate Pipeline (3 gates)
**Each Gate Uses**: Byzantine Consensus (6+ validators)
**Final Approval Uses**: Hierarchical coordination (evaluator authority)

---

## Coordination Pattern Statistics

| Pattern | Skills Using | Avg Agents | Avg Phases | Complexity |
|---------|-------------|-----------|-----------|------------|
| Hierarchical Multi-Phase | 15 | 6.2 | 5-7 | High |
| Parallel Swarm | 12 | 4.5 | 1 | Medium |
| Byzantine Consensus | 4 | 7.8 | 1 | High |
| Research-Then-Action | 10 | 2.0 | 2 | Low |
| Quality Gate Pipeline | 4 | 8.5 | 3 | Very High |

---

## Memory Namespace Conventions

### Hierarchical Multi-Phase
```
swarm/[skill-name]/
├── phase-1/
├── phase-2/
├── ...
└── phase-n/
```

### Parallel Swarm
```
swarm/[skill-name]/
├── parallel-results/
│   ├── agent-1-results
│   ├── agent-2-results
│   └── aggregated
```

### Byzantine Consensus
```
swarm/[skill-name]/
├── consensus/
│   ├── votes/
│   ├── threshold
│   └── decision
```

### Research-Then-Action
```
swarm/[skill-name]/
├── research/
│   ├── findings
│   ├── best-practices
│   └── blueprint
└── implementation/
    └── results
```

### Quality Gate Pipeline
```
quality-gate-1/
├── validators/
└── approval
quality-gate-2/
├── validators/
└── approval
quality-gate-3/
├── validators/
└── approval
```

---

**Generated By**: Claude Code (Sonnet 4.5)
**Date**: 2025-11-02
**Source**: SKILL-AGENT-ASSIGNMENTS.md analysis
**Purpose**: Visual reference for pattern selection and implementation
