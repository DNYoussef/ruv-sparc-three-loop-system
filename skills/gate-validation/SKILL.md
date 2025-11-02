---
name: "Gate Validation"
description: "Validate Deep Research SOP quality gates (Gates 1-3) ensuring rigorous GO/NO-GO decisions based on comprehensive requirement checklists with statistical validation. Use at end of research phases (Foundations→Development→Production) to determine if criteria met for phase transition. Coordinates evaluator agent with ethics-agent, data-steward, and archivist for systematic validation across data quality, baseline replication, novel methods, holistic evaluation, reproducibility, and production readiness. Implements multiple comparison correction (Bonferroni), effect size calculation (Cohen's d), and statistical power analysis (1-β ≥ 0.8)."
pipeline: "Quality Gates (Gates 1, 2, 3)"
quality_gate: "All three gates (1, 2, 3)"
phase: "Phase transitions (1→2→3)"
agents: "evaluator (lead), ethics-agent, data-steward, archivist"
time_estimate: "2-4 hours per gate"
keywords:
  - quality-gate
  - go-no-go-decision
  - gate-validation
  - evaluator-agent
  - phase-transition
  - statistical-rigor
  - multiple-comparison
  - bonferroni-correction
  - cohens-d
  - power-analysis
  - reproducibility-validation
  - ethics-review
---

# Gate Validation

Systematic validation of Deep Research SOP quality gates ensuring rigorous GO/NO-GO decisions at each phase transition.

## Overview

**Purpose**: Validate quality gate requirements and make GO/NO-GO decisions

**When to Use**:
- End of Phase 1 (Foundations) → Gate 1 validation
- End of Phase 2 (Development) → Gate 2 validation
- End of Phase 3 (Production) → Gate 3 validation
- Before phase transitions in research workflow
- When evaluator agent assessment needed

**Quality Gates**:
- **Gate 1**: Data & Methods Validation (baseline replication, data quality, ethics)
- **Gate 2**: Model & Evaluation Validation (novel methods, holistic evaluation, safety)
- **Gate 3**: Production & Artifacts Validation (reproducibility, deployment, archival)

**Prerequisites**:
- Phase work completed (1, 2, or 3)
- All pipeline deliverables available
- Agent coordination active (evaluator, ethics-agent, data-steward, archivist)

**Outputs**:
- Quality gate decision (APPROVED / CONDITIONAL / REJECTED)
- Comprehensive checklist with pass/fail status
- Gap analysis and remediation plan
- Phase transition recommendation
- Memory MCP storage of gate status

**Time Estimate**: 2-4 hours per gate

**Agents Used**: evaluator (lead), ethics-agent, data-steward, archivist

---

## Quick Start

### Gate 1 Validation (End of Phase 1)
```bash
# Validate Gate 1 requirements
claude-code invoke-skill gate-validation --gate 1

# Or use command directly
npx claude-flow@alpha sparc run evaluator \
  "/validate-gate-1 --pipeline F --verbose"
```

### Gate 2 Validation (End of Phase 2)
```bash
# Validate Gate 2 requirements
claude-code invoke-skill gate-validation --gate 2

# Or use command directly
npx claude-flow@alpha sparc run evaluator \
  "/validate-gate-2 --pipeline E --include-ethics"
```

### Gate 3 Validation (End of Phase 3)
```bash
# Validate Gate 3 requirements
claude-code invoke-skill gate-validation --gate 3

# Or use command directly
npx claude-flow@alpha sparc run evaluator \
  "/validate-gate-3 --pipeline G --include-deployment"
```

---

## Detailed Instructions

### Quality Gate 1: Data & Methods Validation

**Objective**: Validate data quality, baseline replication, and ethics before method development

**Requirements Checklist**:

#### 1.1 Literature Review (Pipeline A)
```bash
# Check literature review completeness
npx claude-flow@alpha memory retrieve --key "sop/phase1/literature-review"
```

**Requirements**:
- [ ] ≥50 papers reviewed and cataloged
- [ ] SOTA performance benchmarks identified
- [ ] Research gap analysis documented
- [ ] Hypotheses formulated with testable predictions

**Validation**:
```python
papers_reviewed = len(literature_database.query("status == 'reviewed'"))
assert papers_reviewed >= 50, f"Only {papers_reviewed}/50 papers reviewed"
```

---

#### 1.2 Datasheet (Pipeline B)
```bash
# Check datasheet completion
npx claude-flow@alpha sparc run data-steward \
  "Validate datasheet completeness" \
  --datasheet phase1-foundations/datasheet.md
```

**Requirements**:
- [ ] Datasheet complete (Form F-P1) with ≥90% sections filled
- [ ] Motivation, composition, collection process documented
- [ ] Preprocessing and labeling procedures described
- [ ] Recommended uses and limitations identified
- [ ] Distribution and maintenance plan established

**Validation**:
```python
from datasheet_parser import parse_datasheet

datasheet = parse_datasheet("phase1-foundations/datasheet.md")
completion = datasheet.calculate_completion()
assert completion >= 0.90, f"Datasheet only {completion*100:.1f}% complete"
```

---

#### 1.3 Bias Audit (Pipeline B)
```bash
# Check bias audit results
npx claude-flow@alpha sparc run data-steward \
  "Review bias audit report" \
  --audit-report phase1-foundations/bias-audit.json
```

**Requirements**:
- [ ] Bias audit complete across demographic groups
- [ ] Label distribution analyzed
- [ ] Representation metrics calculated
- [ ] Mitigation strategies documented (if bias detected)
- [ ] Acceptable bias levels (<10% demographic parity gap)

**Validation**:
```python
import json

with open("phase1-foundations/bias-audit.json") as f:
    audit = json.load(f)

max_demographic_gap = max(audit["demographic_parity_differences"].values())
assert max_demographic_gap < 0.10, f"Demographic parity gap {max_demographic_gap*100:.1f}% > 10%"
```

---

#### 1.4 Ethics Review (Pipeline B, F)
```bash
# Validate ethics review
npx claude-flow@alpha sparc run ethics-agent \
  "/assess-risks --component dataset --gate 1 --validation-mode"
```

**Requirements**:
- [ ] Ethics review form (F-F1) complete
- [ ] Risk assessment across 6 domains completed
- [ ] IRB approval obtained (if human subjects data)
- [ ] Dual-use risks assessed and mitigated
- [ ] Ethics review status: APPROVED

**Validation**:
```bash
ethics_status=$(npx claude-flow@alpha memory retrieve --key "sop/gate-1/ethics-status")
if [ "$ethics_status" != "APPROVED" ]; then
    echo "ERROR: Ethics review not approved: $ethics_status"
    exit 1
fi
```

---

#### 1.5 Baseline Replication (Pipeline D)
```bash
# Validate baseline replication
python scripts/validate_baseline_replication.py \
  --baseline-paper "Attention is All You Need" \
  --replicated-checkpoint phase1-foundations/baseline/checkpoint.pth \
  --tolerance 0.01
```

**Requirements**:
- [ ] Baseline implementation complete (100% test coverage)
- [ ] Results within ±1% of published baseline
- [ ] Statistical significance confirmed (paired t-test, p<0.05)
- [ ] Reproducibility tested (3/3 successful runs)
- [ ] Reproducibility package created (Docker + README)

**Validation**:
```python
from scipy import stats

baseline_accuracy = 0.850  # Published
replicated_accuracy = 0.852  # Our implementation
tolerance = 0.01

delta = abs(replicated_accuracy - baseline_accuracy)
assert delta <= tolerance, f"Baseline replication Δ={delta*100:.2f}% > ±1%"

# Statistical test
t_stat, p_value = stats.ttest_rel(baseline_runs, replicated_runs)
assert p_value >= 0.05, f"Significant difference detected (p={p_value:.4f})"
```

---

#### 1.6 Gate 1 Decision
**evaluator agent synthesizes all requirements:**

```bash
npx claude-flow@alpha sparc run evaluator \
  "/validate-gate-1 --pipeline F --comprehensive"
```

**Decision Criteria**:

**APPROVED** (all critical requirements met):
- ✅ Literature review ≥50 papers
- ✅ Datasheet ≥90% complete
- ✅ Bias audit complete, gaps <10%
- ✅ Ethics review APPROVED
- ✅ Baseline replication ±1% tolerance
- ✅ Reproducibility 3/3 runs successful
- **Action**: Proceed to Phase 2 (Method Development)

**CONDITIONAL** (minor gaps, mitigation plan in place):
- ✅ Most requirements met
- ⚠️ Datasheet 85-89% complete (minor sections missing)
- ⚠️ Bias gaps 10-15% (mitigation plan documented)
- **Action**: Proceed to Phase 2 with restrictions, complete missing items in parallel

**REJECTED** (critical issues):
- ❌ Baseline replication >±1% tolerance
- ❌ Ethics review flagged critical risks
- ❌ Reproducibility tests failed
- ❌ Datasheet <85% complete
- **Action**: Return to Phase 1, address critical issues

---

### Quality Gate 2: Model & Evaluation Validation

**Objective**: Validate novel method performance, holistic evaluation, and safety before production

**Requirements Checklist**:

#### 2.1 Novel Method Performance (Pipeline D)
```bash
# Validate novel method vs. baseline
python scripts/validate_method_performance.py \
  --novel phase2-development/novel-method/checkpoint.pth \
  --baseline phase1-foundations/baseline/checkpoint.pth
```

**Requirements**:
- [ ] Novel method implemented with complete codebase
- [ ] Performance meets or exceeds baseline
- [ ] Statistical significance confirmed (p<0.05)
- [ ] Improvement documented (e.g., +2.5% accuracy)
- [ ] Code quality: 100% test coverage, linting passed

**Validation**:
```python
novel_accuracy = 0.875
baseline_accuracy = 0.850

improvement = novel_accuracy - baseline_accuracy
assert improvement >= 0, f"Novel method underperforms baseline by {improvement*100:.1f}%"

t_stat, p_value = stats.ttest_ind(novel_runs, baseline_runs)
assert p_value < 0.05, f"Improvement not statistically significant (p={p_value:.4f})"
```

---

#### 2.2 Ablation Studies (Pipeline D)
```bash
# Validate ablation studies
python scripts/validate_ablations.py \
  --ablation-results phase2-development/ablations/
```

**Requirements**:
- [ ] Minimum 5 components ablated independently
- [ ] 3 runs per ablation configuration
- [ ] Statistical analysis complete (t-tests with Bonferroni correction)
- [ ] Component importance ranking documented
- [ ] Critical components identified (p<0.05)

**Validation**:
```python
from scipy import stats
import numpy as np

ablation_results = load_ablation_results("phase2-development/ablations/")
num_components = len(ablation_results["components"])
assert num_components >= 5, f"Only {num_components}/5 components ablated"

for component in ablation_results["components"]:
    assert len(component["runs"]) >= 3, f"{component['name']}: only {len(component['runs'])}/3 runs"

# Statistical validation with multiple comparison correction
baseline_scores = np.array(ablation_results["baseline_runs"])
p_values = []

for component in ablation_results["components"]:
    ablated_scores = np.array(component["runs"])

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(baseline_scores, ablated_scores)
    p_values.append(p_value)

    # Effect size (Cohen's d)
    mean_diff = np.mean(baseline_scores - ablated_scores)
    pooled_std = np.sqrt((np.std(baseline_scores)**2 + np.std(ablated_scores)**2) / 2)
    cohens_d = mean_diff / pooled_std

    # Statistical power analysis
    from statsmodels.stats.power import ttest_power
    power = ttest_power(cohens_d, nobs=len(baseline_scores), alpha=0.05, alternative='two-sided')

    component["t_statistic"] = t_stat
    component["p_value"] = p_value
    component["cohens_d"] = cohens_d
    component["statistical_power"] = power

    # Validate sufficient power
    if power < 0.8:
        print(f"⚠️  WARNING: {component['name']} has low statistical power ({power:.2f} < 0.80)")

# Multiple comparison correction (Bonferroni)
bonferroni_alpha = 0.05 / num_components
print(f"Bonferroni-corrected α: {bonferroni_alpha:.4f}")

significant_components = []
for component, p_val in zip(ablation_results["components"], p_values):
    if p_val < bonferroni_alpha:
        significant_components.append(component["name"])
        print(f"✅ {component['name']}: Significant after Bonferroni correction (p={p_val:.4f})")
    else:
        print(f"❌ {component['name']}: Not significant after correction (p={p_val:.4f})")

assert len(significant_components) > 0, "No components significant after multiple comparison correction"
```

---

#### 2.3 Holistic Evaluation (Pipeline E)
```bash
# Validate holistic evaluation
python scripts/validate_holistic_evaluation.py \
  --evaluation-report phase2-development/holistic-evaluation/report.md
```

**Requirements**:
- [ ] Evaluation complete across ≥6 dimensions
- [ ] Accuracy: Test accuracy documented
- [ ] Fairness: Demographic parity <10%, equalized odds <10%
- [ ] Robustness: Adversarial robustness tested, OOD detection >0.70 AUROC
- [ ] Efficiency: Latency, throughput, memory within targets
- [ ] Interpretability: SHAP/attention analysis complete
- [ ] Safety: Harmful output rate <0.05%, privacy preserved

**Validation**:
```python
eval_report = parse_evaluation_report("phase2-development/holistic-evaluation/report.md")

# Check dimensions
assert len(eval_report["dimensions"]) >= 6, "Holistic evaluation incomplete"

# Check fairness
assert eval_report["fairness"]["demographic_parity"] < 0.10, "Demographic parity gap >10%"
assert eval_report["fairness"]["equalized_odds_tpr"] < 0.10, "TPR gap >10%"

# Check safety
assert eval_report["safety"]["harmful_output_rate"] < 0.0005, "Harmful output rate >0.05%"
```

---

#### 2.4 Ethics & Safety Review (Pipeline F)
```bash
# Validate Gate 2 ethics review
npx claude-flow@alpha sparc run ethics-agent \
  "/assess-risks --component model --gate 2 --validation-mode"
```

**Requirements**:
- [ ] Ethics review complete for Gate 2
- [ ] Model safety evaluation passed
- [ ] Fairness audit passed
- [ ] Privacy audit passed (membership inference AUC ≤0.55)
- [ ] Adversarial prompt testing passed (>90% rejection)
- [ ] Dual-use risk assessment complete
- [ ] Ethics status: APPROVED

---

#### 2.5 Method Card (Pipeline G)
```bash
# Validate method card
npx claude-flow@alpha sparc run archivist \
  "Validate method card completeness" \
  --method-card phase2-development/method-card.md
```

**Requirements**:
- [ ] Method card ≥90% complete
- [ ] 9 sections filled (Mitchell et al. 2019 template)
- [ ] Performance metrics documented
- [ ] Training details complete
- [ ] Ethical considerations addressed
- [ ] Limitations documented

---

#### 2.6 Gate 2 Decision
**evaluator agent synthesizes all requirements:**

```bash
npx claude-flow@alpha sparc run evaluator \
  "/validate-gate-2 --pipeline E --include-ethics --comprehensive"
```

**Decision Criteria**:

**APPROVED**:
- ✅ Novel method > baseline (statistically significant)
- ✅ Ablation studies ≥5 components
- ✅ Holistic evaluation 6+ dimensions
- ✅ Ethics review APPROVED
- ✅ Method card ≥90% complete
- ✅ Reproducibility 3/3 runs
- **Action**: Proceed to Phase 3 (Production)

**CONDITIONAL**:
- ✅ Most requirements met
- ⚠️ Fairness gaps 10-15% (mitigation plan)
- ⚠️ Adversarial robustness <30% (acceptable for non-adversarial deployment)
- **Action**: Proceed with deployment restrictions

**REJECTED**:
- ❌ Performance regression vs. baseline
- ❌ Ethics review flagged critical safety risks
- ❌ Fairness gaps >15%
- **Action**: Return to Phase 2

---

### Quality Gate 3: Production & Artifacts Validation

**Objective**: Validate reproducibility, deployment readiness, and artifact archival before production

**Requirements Checklist**:

#### 3.1 Model Card (Pipeline G)
```bash
# Validate model card
npx claude-flow@alpha sparc run archivist \
  "Validate model card for Gate 3" \
  --model-card phase3-production/model-card.md
```

**Requirements**:
- [ ] Model card ≥90% complete (Form F-G2)
- [ ] All 9 sections filled (Mitchell et al. 2019)
- [ ] Performance metrics linked
- [ ] Datasheets and risk assessments linked
- [ ] Ethical considerations complete

---

#### 3.2 Reproducibility Package (Pipeline G)
```bash
# Test reproducibility package
python scripts/test_reproducibility.py \
  --package phase3-production/reproducibility-package/ \
  --runs 3
```

**Requirements**:
- [ ] Reproducibility package complete (code + data + environment)
- [ ] Docker environment builds successfully
- [ ] README with ≤5 steps to reproduce
- [ ] All hyperparameters and seeds documented
- [ ] 3/3 reproduction runs successful
- [ ] Results within ±1% of reported values

---

#### 3.3 DOI Assignment (Pipeline G)
```bash
# Validate DOI assignment
npx claude-flow@alpha sparc run archivist \
  "Check DOI assignment status"
```

**Requirements**:
- [ ] Dataset DOI assigned (Zenodo)
- [ ] Model DOI assigned (Zenodo)
- [ ] Code DOI assigned (GitHub + Zenodo)
- [ ] Citation formats generated

---

#### 3.4 Public Registry Publishing (Pipeline G)
```bash
# Validate public registry publishing
python scripts/validate_registry_publishing.py \
  --registries "huggingface,zenodo,mlflow,github"
```

**Requirements**:
- [ ] Code public on GitHub (release created)
- [ ] Model published to HuggingFace Hub
- [ ] Artifacts published to Zenodo
- [ ] Registry URLs documented
- [ ] All artifacts accessible publicly

---

#### 3.5 Deployment Readiness (Pipeline H)
```bash
# Validate deployment readiness
python scripts/validate_deployment_readiness.py \
  --model phase3-production/final-checkpoint.pth \
  --environment production
```

**Requirements**:
- [ ] Deployment plan validated
- [ ] Infrastructure requirements documented
- [ ] Monitoring plan established
- [ ] Incident response plan created
- [ ] Performance benchmarks (production environment) complete
- [ ] Rollback strategy documented

---

#### 3.6 Publication Artifacts (Pipeline I)
```bash
# Validate publication readiness
npx claude-flow@alpha sparc run researcher \
  "Validate publication artifacts" \
  --venue NeurIPS
```

**Requirements**:
- [ ] Research paper draft complete
- [ ] Reproducibility checklist complete (NeurIPS/ICML)
- [ ] Supplementary materials prepared
- [ ] Artifact submission ready (ACM badges)
- [ ] Code release finalized

---

#### 3.7 Gate 3 Decision
**evaluator agent synthesizes all requirements:**

```bash
npx claude-flow@alpha sparc run evaluator \
  "/validate-gate-3 --pipeline G --include-deployment --comprehensive"
```

**Decision Criteria**:

**APPROVED**:
- ✅ Model card ≥90% complete
- ✅ Reproducibility 3/3 runs successful
- ✅ DOIs assigned (dataset, model, code)
- ✅ Code public (GitHub release)
- ✅ Ethics review APPROVED (deployment)
- ✅ Deployment plan validated
- ✅ Publication artifacts ready
- **Action**: DEPLOY to production, submit publication

**CONDITIONAL**:
- ✅ Most requirements met
- ⚠️ Minor documentation gaps (model card 85-89%)
- **Action**: Deploy with documentation fixes in progress

**REJECTED**:
- ❌ Reproducibility tests failed
- ❌ Critical ethics violations
- ❌ DOIs not assigned
- **Action**: Return to Phase 3

---

## Integration with Deep Research SOP

### Quality Gate System
The three quality gates ensure systematic progression through the research lifecycle:

```
Gate 1 (End of Phase 1):
  INPUTS: Literature review, datasheet, bias audit, ethics review, baseline replication
  OUTPUT: GO/NO-GO for method development
  AGENTS: evaluator, data-steward, ethics-agent

Gate 2 (End of Phase 2):
  INPUTS: Novel method, ablations, holistic evaluation, ethics review, method card
  OUTPUT: GO/NO-GO for production deployment
  AGENTS: evaluator, ethics-agent

Gate 3 (End of Phase 3):
  INPUTS: Model card, reproducibility package, DOIs, deployment plan, publication artifacts
  OUTPUT: GO/NO-GO for deployment and publication
  AGENTS: evaluator, archivist
```

---

## Troubleshooting

### Issue: Gate 1 rejected due to baseline replication failure
**Solution**:
```bash
# Debug baseline replication
python scripts/debug_baseline_replication.py \
  --baseline-paper "Attention is All You Need" \
  --verbose

# Check for implementation bugs
python tests/integration/test_gradient_flow.py
python tests/integration/test_numerical_stability.py

# Re-run with stricter determinism
python train_baseline.py --deterministic --seed 42
```

### Issue: Gate 2 rejected due to fairness gaps
**Solution**:
```bash
# Apply fairness mitigation
python scripts/fairness_calibration.py \
  --model phase2-development/novel-method/checkpoint.pth \
  --method "equalized_odds_postprocessing"

# Re-evaluate fairness
python scripts/fairness_eval.py --model calibrated_model.pth
```

### Issue: Gate 3 rejected due to reproducibility failure
**Solution**:
```bash
# Audit reproducibility package
python scripts/audit_reproducibility.py \
  --package phase3-production/reproducibility-package/ \
  --strict --debug

# Fix non-deterministic behavior
python scripts/test_determinism.py --runs 10 --strict
```

---

## Related Skills and Commands

### Related Skills
- `baseline-replication` - Required for Gate 1
- `method-development` - Required for Gate 2
- `holistic-evaluation` - Required for Gate 2
- `reproducibility-audit` - Required for Gate 3
- `deep-research-orchestrator` - Uses this skill for all 3 gates

### Related Commands
- `/validate-gate-1` - Gate 1 validation (evaluator)
- `/validate-gate-2` - Gate 2 validation (evaluator)
- `/validate-gate-3` - Gate 3 validation (evaluator)
- `/assess-risks` - Ethics assessment (ethics-agent)
- `/init-datasheet` - Datasheet creation (data-steward)
- `/init-model-card` - Model card creation (archivist)

---

## References

### Quality Gate Methodologies
- Stage-Gate Process (Cooper, 1990): Phase-gate systems for innovation
- ISO 9001: Quality management systems
- FDA Software Validation Guidance: Pre-market validation requirements

### Academic Standards
- NeurIPS Reproducibility Checklist
- ACM Artifact Evaluation Badging
- IEEE Software Quality Assurance Standards
