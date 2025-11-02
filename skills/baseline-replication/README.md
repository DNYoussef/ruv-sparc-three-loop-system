# Baseline Replication - Silver Tier Documentation

**Skill Tier**: Silver (7+ files)
**Category**: Deep Research SOP - Pipeline D (Method Development)
**Quality Gate**: Gate 1 (Baseline Validation)
**Estimated Time**: 8-12 hours (first baseline), 4-6 hours (subsequent)

---

## Overview

The **Baseline Replication** skill implements ACM Artifact Evaluation standards for reproducing published machine learning baselines with exact reproducibility (±1% tolerance). This is a mandatory prerequisite for developing novel methods in Deep Research SOP Pipeline D.

### Key Features

- **Exact Reproducibility**: ±1% statistical tolerance with deterministic settings
- **ACM Compliance**: Full Artifact Evaluation criteria (Available, Functional, Reproduced)
- **Statistical Validation**: Paired t-tests, confidence intervals, effect size calculation
- **Quality Gate 1**: Complete validation checklist for baseline approval
- **Multi-Agent Coordination**: Researcher, data-steward, coder, tester, archivist, evaluator

---

## Quick Start (30 Minutes)

### 1. Specify Baseline to Replicate

```bash
# Example: BERT on SQuAD 2.0
BASELINE_PAPER="BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2019)"
BASELINE_CODE="https://github.com/google-research/bert"
TARGET_METRIC="Accuracy on SQuAD 2.0"
PUBLISHED_RESULT=0.948
```

### 2. Run Replication Workflow

```bash
./scripts/replicate-baseline.sh \
  --paper "$BASELINE_PAPER" \
  --code "$BASELINE_CODE" \
  --metric "$TARGET_METRIC" \
  --expected "$PUBLISHED_RESULT"
```

### 3. Expected Output

```
✓ Paper analyzed: Extracted 47 hyperparameters
✓ Dataset validated: SQuAD 2.0 matches baseline
✓ Implementation complete: 12 BERT layers, 110M parameters
✓ Training complete: 3 epochs, 26.3 GPU hours
✓ Results validated: 0.945 vs 0.948 (within ±1% tolerance)
✓ Reproducibility verified: 3/3 fresh reproductions successful
→ Quality Gate 1: APPROVED
```

---

## What This Skill Does

### 7-Phase Replication Process

1. **Paper Analysis** (15 min) - Extract methodology, hyperparameters, architecture
2. **Dataset Validation** (20 min) - Verify exact dataset version, splits, preprocessing
3. **Implementation** (2 hours) - Code baseline with deterministic settings
4. **Experiment Execution** (4-8 hours) - Train with monitoring and checkpointing
5. **Results Validation** (30 min) - Statistical comparison with published results
6. **Reproducibility Packaging** (30 min) - Create Docker-based reproduction package
7. **Quality Gate 1 Validation** (15 min) - Evaluate against gate requirements

---

## ACM Compliance

This skill implements **ACM Artifact Review and Badging** standards:

### Artifact Badges

- **Available**: Artifacts publicly accessible with DOI
- **Functional**: Documentation allows independent execution
- **Reproduced**: Independent validation within ±1% tolerance
- **Reusable**: Extensible for novel method development

### Validation Criteria

| Criterion | Requirement | Implementation |
|-----------|-------------|----------------|
| **Completeness** | All code, data, dependencies documented | ✓ Complete reproducibility package |
| **Documentation** | ≤5 steps to reproduce | ✓ Dockerfile + 3-step README |
| **Determinism** | Fixed random seeds, reproducible results | ✓ 3/3 successful reproductions |
| **Tolerance** | Results within ±1% of published | ✓ Statistical validation |

---

## Statistical Validation

### Paired T-Tests

Validate reproduced results against published baselines:

```python
import scipy.stats as stats
import numpy as np

# 3 independent runs
reproduced = [0.945, 0.946, 0.944]
published = 0.948

# Statistical comparison
difference = np.mean(reproduced) - published
percent_diff = (difference / published) * 100

# Within tolerance?
within_tolerance = abs(difference / published) <= 0.01

# T-test
t_stat, p_value = stats.ttest_1samp(reproduced, published)

# Confidence interval
ci_95 = stats.t.interval(0.95, len(reproduced)-1,
                         loc=np.mean(reproduced),
                         scale=stats.sem(reproduced))

print(f"Reproduced: {np.mean(reproduced):.3f} ± {np.std(reproduced):.3f}")
print(f"Published: {published:.3f}")
print(f"Difference: {percent_diff:.2f}%")
print(f"Within ±1%: {within_tolerance}")
print(f"95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
print(f"p-value: {p_value:.4f}")
```

### Effect Size (Cohen's d)

```python
def cohens_d(reproduced, published):
    """Calculate effect size for practical significance"""
    mean_diff = np.mean(reproduced) - published
    pooled_std = np.std(reproduced)
    return mean_diff / pooled_std

d = cohens_d(reproduced, published)
print(f"Cohen's d: {d:.3f}")
# |d| < 0.2: Small effect (acceptable for baseline replication)
```

---

## Integration with Deep Research SOP

### Pipeline D: Method Development Flow

```
Phase 1: Baseline Replication (This Skill)
├── Paper analysis
├── Dataset validation (data-steward)
├── Implementation (coder)
├── Experiment execution (tester)
├── Statistical validation
├── Reproducibility packaging (archivist)
└── Quality Gate 1 approval (evaluator) ✓
    ↓
Phase 2: Novel Method Development
├── Algorithm design
├── Ablation studies (5+ ablations)
├── Statistical testing (Bonferroni correction)
└── Quality Gate 2 submission
    ↓
Phase 3: Production Deployment
```

### Multi-Agent Coordination

```yaml
Sequential Workflow:
  1. researcher:
     - Analyze paper
     - Extract methodology
     - Identify data sources

  2. data-steward:
     - Validate datasets (Form F-C1)
     - Check integrity
     - Verify preprocessing

  3. coder:
     - Implement baseline
     - Add unit tests
     - Code review

  4. tester:
     - Run experiments
     - Monitor training
     - Collect metrics

  5. archivist:
     - Create repro package
     - Test fresh reproduction
     - Archive artifacts

  6. evaluator:
     - Validate Gate 1
     - Generate checklist
     - Approve/Conditional/Reject
```

---

## File Structure

```
baseline-replication/
├── README.md (this file)
├── SKILL.md (main skill definition)
├── examples/
│   ├── example-1-basic-replication.md
│   ├── example-2-statistical-tests.md
│   └── example-3-ablation-studies.md
├── references/
│   ├── acm-compliance.md
│   └── statistical-methods.md
├── graphviz/
│   └── workflow.dot
├── scripts/
│   ├── replicate-baseline.sh
│   ├── analyze-paper.sh
│   ├── validate-dataset.sh
│   ├── implement-baseline.sh
│   ├── run-experiments.sh
│   ├── compare-results.sh
│   ├── create-repro-package.sh
│   ├── test-reproducibility.sh
│   └── validate-gate-1.sh
├── docs/
│   ├── troubleshooting.md
│   ├── best-practices.md
│   └── agent-coordination.md
└── resources/
    └── templates/
        └── bert-base.py
```

---

## Quality Gate 1 Checklist

### Requirements

- [ ] Baseline specification document complete (all hyperparameters documented)
- [ ] Dataset validation passed (exact version, splits, preprocessing)
- [ ] Implementation tested (100% unit test coverage)
- [ ] Results within ±1% of published (statistical validation)
- [ ] Reproducibility package tested in fresh Docker environment (3/3 successful)
- [ ] Documentation complete (≤5 steps to reproduce)
- [ ] All artifacts archived with checksums

### Decision Logic

```python
def evaluate_gate_1(results, reproducibility):
    """Evaluate Quality Gate 1 approval"""

    if results['within_tolerance'] and reproducibility['success_rate'] == 1.0:
        return "APPROVED"  # All requirements met

    elif results['difference_pct'] < 1.5 and reproducibility['success_rate'] >= 0.67:
        return "CONDITIONAL"  # Minor gaps, fixable within 1 week

    else:
        return "REJECT"  # Major gaps, requires significant rework
```

---

## Common Issues and Solutions

### Issue 1: Results Diverge > 1%

**Symptoms**: Reproduced 0.932, published 0.948 (1.7% difference)

**Solution**:
```bash
# Systematic debugging
./scripts/debug-divergence.sh --detailed

# Check 1: Random seeds
python -c "import torch; print(torch.initial_seed())"

# Check 2: Framework version
python -c "import torch; print(torch.__version__)"

# Check 3: Enable deterministic mode
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
```

### Issue 2: Non-Deterministic Results

**Symptoms**: 3 runs produce 0.945, 0.951, 0.938 (high variance)

**Solution**:
```python
# Force deterministic mode
import torch
import random
import numpy as np

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

### Issue 3: Missing Hyperparameters

**Solution**:
```bash
# Check official code config files
grep -r "learning_rate\|batch_size\|warmup" ${BASELINE_CODE}/

# Check GitHub issues
gh issue list --repo ${BASELINE_REPO} --search "hyperparameter"

# Contact authors
./scripts/contact-authors.sh --paper "arXiv:2103.00020"
```

---

## Examples

- [Example 1: Basic Replication (BERT on SQuAD)](examples/example-1-basic-replication.md)
- [Example 2: Statistical Tests and Validation](examples/example-2-statistical-tests.md)
- [Example 3: Ablation Studies](examples/example-3-ablation-studies.md)

---

## References

- [ACM Artifact Evaluation Guidelines](references/acm-compliance.md)
- [Statistical Methods for Baseline Validation](references/statistical-methods.md)
- [ML Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
- [TIER Protocol](https://www.projecttier.org/)

---

## Related Skills

- **method-development** - Develop novel methods after baseline validation
- **holistic-evaluation** - Run HELM + CheckList evaluations (Pipeline E)
- **gate-validation** - Quality Gate approval workflow
- **reproducibility-audit** - Test reproducibility packages
- **literature-synthesis** - PRISMA systematic reviews

---

**Version**: 1.0.0 (Silver Tier)
**Last Updated**: 2025-11-02
**Maintainer**: Deep Research SOP Team
