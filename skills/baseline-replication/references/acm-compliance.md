# ACM Artifact Evaluation Compliance Guide

**Standard**: ACM Artifact Review and Badging (Version 1.1)
**Source**: https://www.acm.org/publications/policies/artifact-review-and-badging-current
**Relevance**: Baseline replication must meet ACM standards for reproducibility

---

## Overview

The **ACM Artifact Review and Badging** initiative provides a standardized framework for evaluating research artifacts (code, data, models). Baseline replication in Deep Research SOP Pipeline D adheres to ACM standards to ensure scientific rigor.

---

## ACM Artifact Badges

### 1. Artifacts Available

**Definition**: Research artifacts are publicly and permanently available.

**Requirements**:
- ✅ Artifacts stored in publicly accessible repository
- ✅ Persistent identifier (DOI) assigned
- ✅ Open-access license (MIT, Apache 2.0, CC BY 4.0)
- ✅ Version control for reproducibility

**Implementation**:
```bash
# Store baseline artifacts in Zenodo with DOI
zenodo upload \
  --file baseline-bert-repro.tar.gz \
  --title "BERT SQuAD 2.0 Baseline Replication" \
  --license "MIT" \
  --assign-doi

# Example DOI: 10.5281/zenodo.1234567
```

**Badge**: ![Artifacts Available](https://img.shields.io/badge/Artifacts-Available-brightgreen)

---

### 2. Artifacts Evaluated - Functional

**Definition**: Artifacts are documented, consistent, complete, and exercisable.

**Requirements**:
- ✅ Complete documentation (≤5 steps to reproduce)
- ✅ All dependencies specified with exact versions
- ✅ Environment reproducibility (Docker/Conda)
- ✅ Scripts and code run without modification
- ✅ Example outputs provided

**Implementation**:
```markdown
# README.md (≤5 steps)

## Quick Reproduction (3 steps)

1. Build environment:
   ```bash
   docker build -t bert-squad:v1.0 .
   ```

2. Download data:
   ```bash
   ./download_squad.sh
   ```

3. Run training:
   ```bash
   docker run --gpus all bert-squad:v1.0 python train.py
   ```

Expected: 0.945 ± 0.001 accuracy (within ±1% of 0.948 published)
Runtime: ~7 hours on 4x V100 GPUs
```

**Verification**:
```bash
# Test in fresh environment
docker pull bert-squad:v1.0
docker run --gpus all bert-squad:v1.0 python train.py --dry-run

# Expected: Successfully loads model, data, runs 1 epoch
```

**Badge**: ![Artifacts Functional](https://img.shields.io/badge/Artifacts-Functional-blue)

---

### 3. Results Reproduced

**Definition**: Independent researchers obtain same results within acceptable tolerance.

**Requirements**:
- ✅ Results within ±1% of published (or domain-specific tolerance)
- ✅ Statistical validation (paired t-tests, confidence intervals)
- ✅ Multiple independent runs (≥3)
- ✅ Documented variance and reproducibility metrics

**Implementation**:
```python
# Statistical validation
import scipy.stats as stats
import numpy as np

# Independent runs by different researchers
researcher_A_runs = [0.945, 0.946, 0.944]
researcher_B_runs = [0.944, 0.947, 0.945]
published_result = 0.948

# Combined validation
all_runs = researcher_A_runs + researcher_B_runs
mean_reproduced = np.mean(all_runs)
std_reproduced = np.std(all_runs, ddof=1)

# Within tolerance?
difference = mean_reproduced - published_result
within_tolerance = abs(difference / published_result) <= 0.01

# Statistical test
t_stat, p_value = stats.ttest_1samp(all_runs, published_result)

print(f"Reproduced: {mean_reproduced:.3f} ± {std_reproduced:.3f}")
print(f"Published:  {published_result:.3f}")
print(f"Difference: {difference:.3f} ({(difference/published_result)*100:.2f}%)")
print(f"Within ±1%: {within_tolerance}")
print(f"p-value: {p_value:.4f}")

# Decision
if within_tolerance and p_value > 0.05:
    print("✓ Results Reproduced badge awarded")
else:
    print("✗ Results do not meet reproduction criteria")
```

**Badge**: ![Results Reproduced](https://img.shields.io/badge/Results-Reproduced-brightgreen)

---

### 4. Artifacts Reusable (Optional, for novel method development)

**Definition**: Artifacts are well-structured, documented, and extensible for future research.

**Requirements**:
- ✅ Modular code architecture
- ✅ Clear API documentation
- ✅ Unit tests with ≥80% coverage
- ✅ Extension examples provided
- ✅ Community-standard code style

**Implementation**:
```bash
# Modular structure
baseline-bert/
├── src/
│   ├── model.py           # Model architecture
│   ├── data_loader.py     # Data loading
│   ├── trainer.py         # Training loop
│   ├── evaluator.py       # Evaluation metrics
│   └── utils.py           # Utilities
├── tests/
│   ├── test_model.py      # Model tests
│   ├── test_data.py       # Data tests
│   └── test_trainer.py    # Training tests
├── examples/
│   ├── extend_model.py    # How to extend architecture
│   └── custom_dataset.py  # How to use custom data
└── README.md              # Documentation
```

**API Documentation**:
```python
def train_baseline(model, train_loader, optimizer, num_epochs):
    """
    Train baseline model with specified configuration.

    Args:
        model: torch.nn.Module, pre-trained BERT model
        train_loader: DataLoader, training data
        optimizer: torch.optim.Optimizer, AdamW optimizer
        num_epochs: int, number of training epochs

    Returns:
        dict: Training metrics (loss, accuracy per epoch)

    Example:
        >>> model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        >>> optimizer = AdamW(model.parameters(), lr=5e-5)
        >>> metrics = train_baseline(model, train_loader, optimizer, num_epochs=3)
        >>> print(metrics['final_accuracy'])
        0.945
    """
    # Implementation...
```

**Badge**: ![Artifacts Reusable](https://img.shields.io/badge/Artifacts-Reusable-orange)

---

## ACM Compliance Checklist for Baseline Replication

### Phase 1: Artifacts Available

- [ ] Upload artifacts to persistent repository (Zenodo, Figshare, GitHub with release)
- [ ] Assign DOI to artifact
- [ ] Apply open-access license (MIT, Apache 2.0, CC BY 4.0)
- [ ] Create CITATION.cff file for proper attribution
- [ ] Document artifact location in paper/documentation

### Phase 2: Artifacts Functional

- [ ] Write ≤5 step reproduction guide
- [ ] Pin all dependency versions (requirements.txt, environment.yml)
- [ ] Create Docker/Conda environment for reproducibility
- [ ] Test reproduction in fresh environment (3+ runs)
- [ ] Provide example outputs for validation

### Phase 3: Results Reproduced

- [ ] Run ≥3 independent reproductions
- [ ] Calculate mean, std, 95% CI
- [ ] Perform paired t-test (p > 0.05)
- [ ] Verify results within ±1% tolerance
- [ ] Document variance and reproducibility metrics

### Phase 4: Artifacts Reusable (Optional)

- [ ] Modularize code architecture
- [ ] Write API documentation with examples
- [ ] Achieve ≥80% unit test coverage
- [ ] Provide extension examples
- [ ] Follow community code style (PEP 8, ESLint)

---

## Validation Workflow

### Step 1: Self-Evaluation

```bash
# Run ACM compliance checker
./scripts/check-acm-compliance.sh \
  --artifacts baseline-bert-repro.tar.gz \
  --checklist acm-compliance-checklist.md
```

**Example Output**:
```
=== ACM Artifact Compliance Check ===

Artifacts Available:
  ✓ Public repository (https://zenodo.org/record/1234567)
  ✓ DOI assigned (10.5281/zenodo.1234567)
  ✓ Open license (MIT)
  ✓ Version control (Git with tags)

Artifacts Functional:
  ✓ Documentation complete (3-step README)
  ✓ Dependencies specified (requirements.txt)
  ✓ Environment reproducible (Dockerfile)
  ✓ Runs without modification (3/3 successful)
  ✓ Example outputs provided

Results Reproduced:
  ✓ Within ±1% tolerance (0.945 vs 0.948)
  ✓ Statistical validation (p = 0.0955)
  ✓ Multiple runs (3 independent runs)
  ✓ Variance documented (0.001 std)

Artifacts Reusable:
  ✓ Modular architecture
  ✓ API documentation
  ✓ 95% test coverage
  ✓ Extension examples
  ✓ PEP 8 compliant

=== All ACM badges awarded ===
```

### Step 2: Independent Validation

**Reviewer Instructions**:
```markdown
# Independent Reviewer Checklist

## Artifacts Available
- [ ] Access artifacts via DOI link
- [ ] Verify license allows reuse
- [ ] Check version control history

## Artifacts Functional
- [ ] Follow ≤5 step instructions
- [ ] Build environment successfully
- [ ] Run training script
- [ ] Compare outputs with provided examples

## Results Reproduced
- [ ] Run 3 independent experiments
- [ ] Calculate mean ± std
- [ ] Compare with published result
- [ ] Verify within ±1% tolerance

## Artifacts Reusable (Optional)
- [ ] Review code structure
- [ ] Run unit tests
- [ ] Attempt to extend/modify baseline
- [ ] Check documentation quality
```

### Step 3: Badge Application

```bash
# Apply for ACM badges
./scripts/apply-acm-badges.sh \
  --paper "BERT Baseline Replication" \
  --doi "10.5281/zenodo.1234567" \
  --badges "Available,Functional,Reproduced"
```

---

## Common Issues and Solutions

### Issue 1: Results Do Not Reproduce (>1% difference)

**Symptoms**: Mean reproduced = 0.932, published = 0.948 (1.7% difference)

**Solution**:
```bash
# Debug systematically
./scripts/debug-divergence.sh \
  --reproduced 0.932 \
  --published 0.948 \
  --tolerance 0.01

# Common causes:
# 1. Random seed not set
# 2. Framework version mismatch (PyTorch 1.7 vs 1.13)
# 3. Hardware differences (V100 vs A100 numerical precision)
# 4. Missing hyperparameter (learning rate schedule)
# 5. Dataset preprocessing difference
```

### Issue 2: Environment Not Reproducible

**Symptoms**: Docker build fails, dependency conflicts

**Solution**:
```dockerfile
# Pin ALL versions in Dockerfile
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN pip install --no-cache-dir \
    transformers==4.24.0 \
    datasets==2.7.1 \
    scipy==1.9.3 \
    numpy==1.23.5

# Test build
docker build -t bert-squad:v1.0 .
docker run --rm bert-squad:v1.0 python -c "import torch; print(torch.__version__)"
# Expected: 1.13.1+cu116
```

### Issue 3: Documentation Insufficient

**Symptoms**: Reviewers cannot reproduce results due to missing steps

**Solution**:
```markdown
# Detailed README with ALL steps

## Prerequisites
- Docker 20.10+ with GPU support
- NVIDIA Driver 470+ (for CUDA 11.6)
- 16GB+ GPU memory (V100, A100, RTX 3090)
- 50GB disk space

## Step-by-Step Reproduction

1. Clone repository:
   ```bash
   git clone https://zenodo.org/record/1234567/bert-baseline
   cd bert-baseline
   ```

2. Build Docker environment (15 minutes):
   ```bash
   docker build -t bert-squad:v1.0 .
   ```

3. Download SQuAD 2.0 dataset (2 minutes):
   ```bash
   ./download_squad.sh
   ```

4. Run training (7 hours on 4x V100):
   ```bash
   docker run --gpus all -v $(pwd):/workspace bert-squad:v1.0 \
     python train.py --config config/bert-squad.yaml
   ```

5. Evaluate results:
   ```bash
   python evaluate.py --checkpoint checkpoints/best-model.pt
   ```

Expected output:
```
Accuracy: 0.945 ± 0.001 (within ±1% of 0.948 published)
```
```

---

## Summary

### ACM Badge Requirements

| Badge | Core Requirements | Verification |
|-------|-------------------|--------------|
| **Available** | Public + DOI + License | ✓ Zenodo/Figshare |
| **Functional** | Docs + Env + Runs | ✓ Fresh Docker test |
| **Reproduced** | ±1% + Stats + 3 runs | ✓ T-test validation |
| **Reusable** | Modular + Docs + Tests | ✓ Extension examples |

### Quality Gate 1 Integration

**Baseline replication Quality Gate 1 approval requires**:
- ✅ **Artifacts Available** badge (mandatory)
- ✅ **Artifacts Functional** badge (mandatory)
- ✅ **Results Reproduced** badge (mandatory)
- ⚪ **Artifacts Reusable** badge (optional, but recommended for method development)

---

## Resources

- [ACM Artifact Review and Badging](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
- [ACM Artifact Badging FAQ](https://www.acm.org/publications/policies/artifact-review-and-badging-faq)
- [ACM Reproducibility Badge Guidelines](https://www.acm.org/publications/artifacts)
- [ML Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
- [TIER Protocol](https://www.projecttier.org/)

---

**Version**: ACM Badging Standard v1.1
**Last Updated**: 2025-11-02
**Maintained By**: Deep Research SOP Team
