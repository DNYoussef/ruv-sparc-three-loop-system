# Deep Research SOP - Gap Analysis & Implementation Plan

**Date:** 2025-11-01
**Version:** 1.0
**Status:** Initial Analysis

---

## EXECUTIVE SUMMARY

This document provides a comprehensive gap analysis between the **Unified AI Deep Research Pipeline SOP** and the current Claude Code system capabilities. The analysis identifies missing commands, agents, and skills required to fully implement the SOP's research integrity framework.

### Current State
- **Commands:** 4 (in .claude/commands/)
- **Agents:** 54+ documented in CLAUDE.md
- **Skills:** 12 (in .claude/skills/)
- **Graphviz Diagrams:** 286 total (recently added 15 RE components)

### SOP Requirements
- **Pipelines:** 9 (A-I)
- **Agent Roles:** 8 core roles
- **Quality Gates:** 3 mandatory checkpoints
- **Forms/Templates:** 15+ required artifacts
- **Phases:** 3 major phases + ongoing deployment

### Gap Summary
- **Missing Commands:** ~45 specialized research commands
- **Missing Agents:** ~12 specialized research agents
- **Missing Skills:** ~9 end-to-end research workflow skills
- **Missing Integrations:** PRISMA, HELM, CheckList, ML Test Score

---

## TABLE OF CONTENTS

1. [Top-Down Gap Analysis (SOP → System)](#1-top-down-gap-analysis)
2. [Bottom-Up Gap Analysis (System → SOP)](#2-bottom-up-gap-analysis)
3. [MECE Framework Mapping](#3-mece-framework-mapping)
4. [Priority Matrix](#4-priority-matrix)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Component Specifications](#6-component-specifications)
7. [Integration Requirements](#7-integration-requirements)

---

## 1. TOP-DOWN GAP ANALYSIS (SOP → System)

### 1.1 Pipeline A: Systematic Literature & Gap-Finding

**SOP Requirements:**
- PRISMA protocol design and execution
- Multi-database literature search
- Systematic screening (title/abstract → full-text)
- Inter-rater reliability calculation
- Evidence synthesis and gap mapping

**Current Capabilities:**
- ✅ `researcher` agent (general research)
- ✅ WebFetch tool (web content retrieval)
- ❌ No PRISMA-specific workflow
- ❌ No systematic screening process
- ❌ No inter-rater reliability calculation

**Missing Components:**

| Type | Name | Purpose |
|------|------|---------|
| **Command** | `/prisma-init` | Initialize PRISMA protocol |
| **Command** | `/literature-search` | Execute multi-database searches |
| **Command** | `/screen-abstracts` | Systematic abstract screening |
| **Command** | `/calculate-kappa` | Inter-rater reliability (Cohen's κ) |
| **Command** | `/synthesize-evidence` | Evidence synthesis and gap mapping |
| **Agent** | `systematic-reviewer` | PRISMA workflow orchestration |
| **Agent** | `literature-screener` | Abstract/full-text screening |
| **Skill** | `systematic-literature-review` | End-to-end PRISMA workflow |

---

### 1.2 Pipeline B: Baseline Replication

**SOP Requirements:**
- Baseline identification and selection
- Environment specification (Docker, conda)
- Multi-seed experimental execution
- Variance analysis and statistical testing
- Replication status determination

**Current Capabilities:**
- ✅ `coder` agent (implementation)
- ✅ `tester` agent (validation)
- ✅ Bash tool (script execution)
- ❌ No replication-specific workflow
- ❌ No variance analysis automation

**Missing Components:**

| Type | Name | Purpose |
|------|------|---------|
| **Command** | `/identify-baselines` | Select replication targets |
| **Command** | `/specify-environment` | Generate reproducible env specs |
| **Command** | `/run-multi-seed` | Execute experiments across seeds |
| **Command** | `/analyze-variance` | Calculate mean ± CI, effect sizes |
| **Command** | `/test-replication` | Statistical replication validation |
| **Agent** | `replication-specialist` | Baseline replication orchestration |
| **Agent** | `statistical-analyst` | Variance and hypothesis testing |
| **Skill** | `baseline-replication` | Full replication workflow |

---

### 1.3 Pipeline C: Data-Centric Build

**SOP Requirements:**
- Data inventory and access management
- Datasheet for Datasets completion
- Data splits specification (fixed seeds)
- Bias audit (protected attributes)
- Data versioning (DVC, Git LFS)

**Current Capabilities:**
- ✅ File operations (Read, Write, Edit)
- ✅ Bash tool (data processing scripts)
- ❌ No Datasheet template/workflow
- ❌ No bias audit automation
- ❌ No data versioning integration

**Missing Components:**

| Type | Name | Purpose |
|------|------|---------|
| **Command** | `/init-datasheet` | Initialize Datasheet template |
| **Command** | `/audit-bias` | Automated bias analysis |
| **Command** | `/create-splits` | Generate fixed train/val/test splits |
| **Command** | `/version-data` | DVC/Git LFS integration |
| **Command** | `/analyze-distribution` | Data distribution analysis |
| **Agent** | `data-steward` | Data quality and governance |
| **Agent** | `bias-auditor` | Fairness and bias analysis |
| **Skill** | `data-centric-build` | Complete data curation workflow |

---

### 1.4 Pipeline D: Method Development & Ablations

**SOP Requirements:**
- Hypothesis definition and formalization
- Modular component implementation
- Ablation matrix design (MECE coverage)
- Multi-seed ablation execution
- Component contribution analysis with CIs

**Current Capabilities:**
- ✅ `coder` agent (implementation)
- ✅ `planner` agent (planning)
- ❌ No ablation-specific workflow
- ❌ No hypothesis testing framework

**Missing Components:**

| Type | Name | Purpose |
|------|------|---------|
| **Command** | `/define-hypotheses` | Formalize testable hypotheses |
| **Command** | `/design-ablation-matrix` | Create MECE ablation grid |
| **Command** | `/run-ablations` | Execute all ablation configurations |
| **Command** | `/analyze-components` | Calculate Δmean ± CI per component |
| **Command** | `/test-hypotheses` | Statistical hypothesis testing |
| **Agent** | `methodologist` | Method development and ablations |
| **Agent** | `ablation-analyst` | Ablation analysis specialist |
| **Skill** | `method-development` | Full method dev + ablations workflow |

---

### 1.5 Pipeline E: Holistic Evaluation

**SOP Requirements:**
- HELM evaluation (8 metric dimensions × tasks)
- CheckList test suite (MFT, INV, DIR)
- Multi-metric dashboard generation
- Statistical validation of all claims
- Effect size calculation (Cohen's d)

**Current Capabilities:**
- ✅ `tester` agent (testing)
- ✅ `reviewer` agent (review)
- ❌ No HELM integration
- ❌ No CheckList integration
- ❌ No multi-metric evaluation framework

**Missing Components:**

| Type | Name | Purpose |
|------|------|---------|
| **Command** | `/helm-evaluate` | HELM grid execution |
| **Command** | `/checklist-generate` | Create CheckList test suite |
| **Command** | `/checklist-run` | Execute CheckList tests |
| **Command** | `/calculate-effect-size` | Cohen's d and other effect sizes |
| **Command** | `/validate-claims` | Statistical claim validation |
| **Agent** | `evaluator` | Holistic evaluation orchestration |
| **Agent** | `metrics-analyst` | Multi-metric analysis |
| **Skill** | `holistic-evaluation` | Complete HELM + CheckList workflow |

---

### 1.6 Pipeline F: Safety & Ethics

**SOP Requirements:**
- Multi-dimensional risk assessment
- Red-teaming protocol execution
- Domain-specific compliance (CONSORT-AI, etc.)
- Mitigation plan development
- Impact statement generation

**Current Capabilities:**
- ✅ `reviewer` agent (code review)
- ❌ No ethics/safety framework
- ❌ No red-teaming capability
- ❌ No risk assessment tools

**Missing Components:**

| Type | Name | Purpose |
|------|------|---------|
| **Command** | `/assess-risks` | Multi-dimensional risk assessment |
| **Command** | `/red-team` | Execute red-teaming attacks |
| **Command** | `/check-compliance` | Domain compliance verification |
| **Command** | `/generate-impact-statement` | Create impact assessment |
| **Command** | `/plan-mitigations` | Develop risk mitigation plans |
| **Agent** | `ethics-agent` | Safety and ethics orchestration |
| **Agent** | `red-team-specialist` | Adversarial testing |
| **Agent** | `compliance-officer` | Regulatory compliance |
| **Skill** | `safety-and-ethics` | Complete safety/ethics workflow |

---

### 1.7 Pipeline G: Reproducibility & Artifacts

**SOP Requirements:**
- Model Card completion
- Datasheet finalization
- Reproducibility checklist (NeurIPS)
- External reproducibility testing
- Artifact packaging (DOI, Zenodo)

**Current Capabilities:**
- ✅ `coder` agent (code organization)
- ✅ Git operations (Bash tool)
- ❌ No Model Card template
- ❌ No reproducibility checklist workflow
- ❌ No external testing framework

**Missing Components:**

| Type | Name | Purpose |
|------|------|---------|
| **Command** | `/init-model-card` | Initialize Model Card |
| **Command** | `/complete-reproducibility-checklist` | NeurIPS checklist |
| **Command** | `/package-artifacts` | Create release package |
| **Command** | `/test-reproducibility` | External reproduction test |
| **Command** | `/create-doi` | Zenodo/Figshare DOI creation |
| **Agent** | `archivist` | Artifact packaging orchestration |
| **Agent** | `reproducibility-tester` | External testing specialist |
| **Skill** | `reproducibility-artifacts` | Complete artifact workflow |

---

### 1.8 Pipeline H: Production Readiness

**SOP Requirements:**
- ML Test Score calculation (0-10 scale)
- Tech-debt identification and tracking
- Monitoring infrastructure setup
- Incident response planning
- Deployment validation

**Current Capabilities:**
- ✅ `cicd-engineer` agent (CI/CD)
- ✅ Bash tool (infrastructure scripts)
- ❌ No ML Test Score framework
- ❌ No tech-debt tracking
- ❌ No monitoring setup automation

**Missing Components:**

| Type | Name | Purpose |
|------|------|---------|
| **Command** | `/calculate-ml-test-score` | ML Test Score rubric |
| **Command** | `/audit-tech-debt` | Tech-debt identification |
| **Command** | `/setup-monitoring` | Prometheus/Grafana setup |
| **Command** | `/create-incident-plan` | Incident response runbook |
| **Command** | `/validate-deployment` | Pre-deployment validation |
| **Agent** | `reliability-engineer` | Production readiness orchestration |
| **Agent** | `monitoring-specialist` | Monitoring and alerting |
| **Skill** | `production-readiness` | Full deployment workflow |

---

### 1.9 Pipeline I: Anti-Cascade Governance

**SOP Requirements:**
- Stakeholder mapping
- Cascade risk identification
- Guardrail implementation
- Socio-technical quality control
- Feedback loop monitoring

**Current Capabilities:**
- ❌ No cascade governance framework
- ❌ No stakeholder analysis tools

**Missing Components:**

| Type | Name | Purpose |
|------|------|---------|
| **Command** | `/map-stakeholders` | Stakeholder analysis |
| **Command** | `/identify-cascades` | Cascade risk mapping |
| **Command** | `/implement-guardrails` | Quality control guardrails |
| **Command** | `/monitor-feedback-loops` | Feedback loop detection |
| **Agent** | `governance-specialist` | Anti-cascade orchestration |
| **Skill** | `anti-cascade-governance` | Complete governance workflow |

---

## 2. BOTTOM-UP GAP ANALYSIS (System → SOP)

### 2.1 Existing Skills Assessment

**Current Skills (12 in .claude/skills/):**
1. Unknown - requires directory listing

**SOP Alignment:**
- Need to map existing skills to SOP pipelines
- Identify which existing skills can be enhanced vs. new creation needed

### 2.2 Existing Agents Assessment

**Current Agents (54+ documented):**
- Core: coder, reviewer, tester, planner, researcher
- Swarm: hierarchical-coordinator, mesh-coordinator, adaptive-coordinator
- Consensus: byzantine-coordinator, raft-manager, gossip-coordinator
- Performance: perf-analyzer, performance-benchmarker
- GitHub: github-modes, pr-manager, code-review-swarm, issue-tracker
- SPARC: sparc-coord, sparc-coder, specification, pseudocode, architecture
- Specialized: backend-dev, mobile-dev, ml-developer, cicd-engineer
- Reverse Engineering: RE-binary-analyst, RE-decompiler, RE-malware-analyst, RE-vulnerability-hunter, RE-code-reconstructor

**SOP Alignment:**
| Existing Agent | Maps to SOP Role | Usable As-Is? |
|----------------|-----------------|---------------|
| `planner` | Planner Agent | ✅ Yes (enhance with PRISMA) |
| `researcher` | Fetcher/Query Agent | ⚠️ Partial (add RL optimization) |
| `coder` | — | ✅ Yes (supporting role) |
| `tester` | — | ✅ Yes (supporting role) |
| `reviewer` | — | ✅ Yes (supporting role) |
| `ml-developer` | Methodologist Agent | ⚠️ Partial (add ablation workflows) |
| `cicd-engineer` | Reliability Agent | ⚠️ Partial (add ML Test Score) |
| — | Data Steward Agent | ❌ Missing |
| — | Evaluator Agent | ❌ Missing |
| — | Ethics Agent | ❌ Missing |
| — | Archivist Agent | ❌ Missing |

**Gap:** 4 critical SOP agents completely missing, 3 existing agents need enhancement

---

## 3. MECE FRAMEWORK MAPPING

### 3.1 Command Categories (MECE)

Using MECE (Mutually Exclusive, Collectively Exhaustive) framework:

```
ALL RESEARCH COMMANDS
├── Literature Management (6)
│   ├── /prisma-init
│   ├── /literature-search
│   ├── /screen-abstracts
│   ├── /calculate-kappa
│   ├── /synthesize-evidence
│   └── /update-prisma-diagram
├── Experimentation (10)
│   ├── /identify-baselines
│   ├── /specify-environment
│   ├── /run-multi-seed
│   ├── /analyze-variance
│   ├── /test-replication
│   ├── /define-hypotheses
│   ├── /design-ablation-matrix
│   ├── /run-ablations
│   ├── /analyze-components
│   └── /test-hypotheses
├── Data Management (5)
│   ├── /init-datasheet
│   ├── /audit-bias
│   ├── /create-splits
│   ├── /version-data
│   └── /analyze-distribution
├── Evaluation (5)
│   ├── /helm-evaluate
│   ├── /checklist-generate
│   ├── /checklist-run
│   ├── /calculate-effect-size
│   └── /validate-claims
├── Ethics & Safety (5)
│   ├── /assess-risks
│   ├── /red-team
│   ├── /check-compliance
│   ├── /generate-impact-statement
│   └── /plan-mitigations
├── Artifacts & Reproducibility (5)
│   ├── /init-model-card
│   ├── /complete-reproducibility-checklist
│   ├── /package-artifacts
│   ├── /test-reproducibility
│   └── /create-doi
├── Production & Deployment (5)
│   ├── /calculate-ml-test-score
│   ├── /audit-tech-debt
│   ├── /setup-monitoring
│   ├── /create-incident-plan
│   └── /validate-deployment
└── Governance (4)
    ├── /map-stakeholders
    ├── /identify-cascades
    ├── /implement-guardrails
    └── /monitor-feedback-loops
```

**Total New Commands Needed:** 45

---

### 3.2 Agent Categories (MECE)

```
ALL RESEARCH AGENTS
├── Core SOP Roles (8)
│   ├── Planner Agent (✅ exists, enhance)
│   ├── Fetcher/Query Agent (⚠️ partial: researcher)
│   ├── Methodologist Agent (❌ NEW)
│   ├── Data Steward Agent (❌ NEW)
│   ├── Evaluator Agent (❌ NEW)
│   ├── Ethics Agent (❌ NEW)
│   ├── Archivist Agent (❌ NEW)
│   └── Reliability Agent (⚠️ partial: cicd-engineer)
├── Specialized Support (6)
│   ├── systematic-reviewer
│   ├── literature-screener
│   ├── replication-specialist
│   ├── statistical-analyst
│   ├── bias-auditor
│   └── ablation-analyst
└── Advanced Specialists (5)
    ├── metrics-analyst
    ├── red-team-specialist
    ├── compliance-officer
    ├── reproducibility-tester
    └── governance-specialist
```

**Total New Agents Needed:** 12 (4 critical + 8 supporting)

---

### 3.3 Skill Categories (MECE)

```
ALL RESEARCH SKILLS
├── Literature & Planning (1)
│   └── systematic-literature-review
├── Experimentation (2)
│   ├── baseline-replication
│   └── method-development
├── Data & Ethics (2)
│   ├── data-centric-build
│   └── safety-and-ethics
├── Evaluation & Validation (1)
│   └── holistic-evaluation
├── Reproducibility (1)
│   └── reproducibility-artifacts
├── Production (1)
│   └── production-readiness
└── Governance (1)
    └── anti-cascade-governance
```

**Total New Skills Needed:** 9

---

## 4. PRIORITY MATRIX

### 4.1 Critical Path Analysis

Using Eisenhower Matrix (Urgent/Important):

| Priority | Component Type | Name | Reason |
|----------|---------------|------|--------|
| **P0 (Critical)** | Agent | `data-steward` | Required for Gate 1 |
| **P0 (Critical)** | Agent | `evaluator` | Required for Gate 2 |
| **P0 (Critical)** | Agent | `archivist` | Required for Gate 3 |
| **P0 (Critical)** | Agent | `ethics-agent` | Required for all gates |
| **P0 (Critical)** | Command | `/init-datasheet` | Foundation for data work |
| **P0 (Critical)** | Command | `/helm-evaluate` | Foundation for evaluation |
| **P0 (Critical)** | Command | `/init-model-card` | Foundation for artifacts |
| **P1 (High)** | Agent | `methodologist` | Pipeline D critical |
| **P1 (High)** | Agent | `systematic-reviewer` | Pipeline A critical |
| **P1 (High)** | Command | `/prisma-init` | Literature workflows |
| **P1 (High)** | Command | `/run-ablations` | Method development |
| **P2 (Medium)** | Skill | `systematic-literature-review` | End-to-end workflow |
| **P2 (Medium)** | Skill | `holistic-evaluation` | End-to-end workflow |
| **P3 (Low)** | Agent | `governance-specialist` | Pipeline I |
| **P3 (Low)** | Command | `/monitor-feedback-loops` | Advanced feature |

---

## 5. IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Week 1-2)
**Goal:** Critical agents + essential commands for Gate 1

**Deliverables:**
1. **Agents (P0):**
   - `data-steward`
   - `ethics-agent`
   - `archivist`
   - `evaluator`

2. **Commands (Essential):**
   - `/init-datasheet`
   - `/prisma-init`
   - `/assess-risks`
   - `/init-model-card`

3. **Documentation:**
   - Agent YAML files
   - Command markdown files
   - Graphviz diagrams (4 agents + 4 commands)
   - CLAUDE.md updates

**Success Criteria:**
- ✅ All P0 agents created and tested
- ✅ Essential commands functional
- ✅ Documentation complete

---

### Phase 2: Experimentation (Week 3-4)
**Goal:** Method development and evaluation capabilities

**Deliverables:**
1. **Agents (P1):**
   - `methodologist`
   - `statistical-analyst`
   - `metrics-analyst`

2. **Commands (Experimentation):**
   - `/run-multi-seed`
   - `/analyze-variance`
   - `/run-ablations`
   - `/helm-evaluate`
   - `/checklist-run`

3. **Skills (Core):**
   - `baseline-replication`
   - `method-development`
   - `holistic-evaluation`

**Success Criteria:**
- ✅ Ablation workflows functional
- ✅ HELM integration complete
- ✅ 3 core skills operational

---

### Phase 3: Literature & Data (Week 5-6)
**Goal:** Complete literature and data pipelines

**Deliverables:**
1. **Agents:**
   - `systematic-reviewer`
   - `literature-screener`
   - `bias-auditor`

2. **Commands (Literature):**
   - `/literature-search`
   - `/screen-abstracts`
   - `/calculate-kappa`
   - `/synthesize-evidence`

3. **Commands (Data):**
   - `/audit-bias`
   - `/create-splits`
   - `/version-data`

4. **Skills:**
   - `systematic-literature-review`
   - `data-centric-build`

**Success Criteria:**
- ✅ PRISMA workflow complete
- ✅ Bias auditing functional
- ✅ 2 additional skills operational

---

### Phase 4: Production & Governance (Week 7-8)
**Goal:** Complete production and governance pipelines

**Deliverables:**
1. **Agents:**
   - `reliability-engineer` (enhance cicd-engineer)
   - `monitoring-specialist`
   - `governance-specialist`

2. **Commands (Production):**
   - `/calculate-ml-test-score`
   - `/setup-monitoring`
   - `/validate-deployment`

3. **Commands (Governance):**
   - `/map-stakeholders`
   - `/identify-cascades`
   - `/implement-guardrails`

4. **Skills:**
   - `production-readiness`
   - `anti-cascade-governance`

**Success Criteria:**
- ✅ ML Test Score framework operational
- ✅ All 9 pipelines complete
- ✅ Full SOP compliance

---

## 6. COMPONENT SPECIFICATIONS

### 6.1 Agent: data-steward

**Priority:** P0 (Critical)
**Pipeline:** C (Data-Centric Build)
**Dependencies:** None

**Responsibilities:**
1. Data inventory management
2. Datasheet completion (Form F-C1)
3. Bias audit execution
4. Data versioning (DVC/Git LFS)
5. Data splits specification
6. Quality gate approval (Gate 1)

**Commands Used:**
- `/init-datasheet`
- `/audit-bias`
- `/create-splits`
- `/version-data`
- `/analyze-distribution`

**Triggers:**
- User mentions "data", "dataset", "Datasheet"
- Pipeline C invoked
- Gate 1 preparation

**Integration Points:**
- Memory MCP: Store Datasheet metadata
- Connascence MCP: Data quality checks
- Git: Version control for data manifests

**Graphviz Diagram:** `agents/data-steward-process.dot`

---

### 6.2 Command: /init-datasheet

**Priority:** P0 (Critical)
**Category:** Data Management
**Agent:** data-steward

**Purpose:** Initialize Datasheet for Datasets (Form F-C1)

**Syntax:**
```bash
/init-datasheet <dataset-name> [--template=<type>]
```

**Workflow:**
1. Create datasheet template (sections: Motivation, Composition, Collection, etc.)
2. Extract dataset metadata (size, format, license)
3. Initialize bias audit section
4. Create version control entry
5. Store in Memory MCP with project context

**Output:**
- `docs/DATASHEET_<dataset-name>.md`
- Memory MCP entry with tags: project, data, datasheet
- Initialization confirmation

**Graphviz Diagram:** `commands/init-datasheet-process.dot`

---

### 6.3 Skill: systematic-literature-review

**Priority:** P2 (Medium)
**Pipeline:** A (Systematic Literature & Gap-Finding)
**Agents:** systematic-reviewer, literature-screener, planner

**Phases:**
1. **Research Question:** PICO framework (F-A2)
2. **PRISMA Protocol:** Design and registration (F-A1)
3. **Database Search:** Multi-source execution
4. **Screening:** Title/abstract → full-text
5. **Synthesis:** Evidence synthesis and gap map
6. **Approval:** Gate 1 preparation

**Commands Orchestrated:**
1. `/prisma-init`
2. `/literature-search`
3. `/screen-abstracts`
4. `/calculate-kappa`
5. `/synthesize-evidence`

**Triggers:**
- "systematic review", "literature review", "PRISMA"
- Pipeline A invoked

**Deliverables:**
- Completed F-A1 (PRISMA Protocol)
- PRISMA flow diagram
- Evidence synthesis report
- Gap map

**Graphviz Diagram:** `skills/systematic-literature-review-process.dot`

---

## 7. INTEGRATION REQUIREMENTS

### 7.1 Memory MCP Integration

**Purpose:** Persistent storage of SOP artifacts across sessions

**Required Memory Keys:**
```
sop/
├── projects/<project-id>/
│   ├── phase
│   ├── gates/
│   │   ├── gate1-status
│   │   ├── gate2-status
│   │   └── gate3-status
│   ├── pipelines/
│   │   ├── pipeline-a-prisma
│   │   ├── pipeline-b-replication
│   │   ├── pipeline-c-data
│   │   ├── pipeline-d-method
│   │   ├── pipeline-e-evaluation
│   │   ├── pipeline-f-ethics
│   │   ├── pipeline-g-artifacts
│   │   ├── pipeline-h-production
│   │   └── pipeline-i-governance
│   └── forms/
│       ├── f-a1-prisma
│       ├── f-a2-research-question
│       ├── f-c1-datasheet
│       ├── f-d1-ablation-matrix
│       └── ...
```

**Tagging Protocol:**
All SOP writes must include:
- **WHO:** Agent name (data-steward, evaluator, etc.)
- **WHEN:** ISO timestamp
- **PROJECT:** Research project ID
- **WHY:** Intent (gate-prep, pipeline-execution, artifact-completion)
- **PIPELINE:** Which SOP pipeline (A-I)
- **PHASE:** Which phase (1-3)

---

### 7.2 Connascence MCP Integration

**Purpose:** Code quality checks for research code

**Integration Points:**
1. **Baseline Replication:** Analyze baseline code for quality issues
2. **Method Development:** Detect coupling in ablation code
3. **Production Readiness:** ML Test Score code quality dimension

**Usage:**
```javascript
// In reliability-engineer agent
const quality = await connascence.analyze_file('train.py');
if (quality.violations.length > 0) {
  mlTestScore.code_quality = 0;  // Fail quality check
}
```

---

### 7.3 External Tool Integration

**Required:**
1. **HELM Toolkit:** `pip install helm-toolkit`
2. **CheckList:** `pip install checklist-toolkit`
3. **DVC:** `pip install dvc`
4. **MLflow:** `pip install mlflow`
5. **Weights & Biases:** `pip install wandb`

**Installation Command:**
```bash
/install-sop-tools
```

---

## 8. NEXT STEPS

### Immediate Actions (This Session)
1. ✅ Complete gap analysis (THIS DOCUMENT)
2. ⬜ Create first P0 agent: `data-steward`
3. ⬜ Create first P0 command: `/init-datasheet`
4. ⬜ Generate Graphviz diagrams for above
5. ⬜ Update CLAUDE.md with new components

### Phase 1 Completion Checklist
- ⬜ 4 P0 agents created
- ⬜ 4 essential commands created
- ⬜ 8 Graphviz diagrams generated
- ⬜ CLAUDE.md updated with triggers
- ⬜ Documentation ingested into Memory MCP

### Long-term Success Criteria
- ⬜ All 9 SOP pipelines implemented
- ⬜ All 3 quality gates functional
- ⬜ Full integration with Memory + Connascence MCPs
- ⬜ External tool integration complete
- ⬜ End-to-end research workflow validated

---

**END OF GAP ANALYSIS**

**Next Document:** `deep-research-sop-implementation-phase1.md`
