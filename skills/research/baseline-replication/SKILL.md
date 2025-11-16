---
name: baseline-replication
description: "Replicate published ML baseline experiments with exact reproducibility\
  \ (\xB11% tolerance) for Deep Research SOP Pipeline D. Use when validating baselines,\
  \ reproducing experiments, verifying published results, or preparing for novel method\
  \ development."
version: 1.0.0
category: research
tags:
- research
- analysis
- planning
author: ruv
---

# Baseline Replication

## Overview
Replicates published machine learning baseline methods with exact reproducibility, ensuring results match within ±1% tolerance. This skill implements Deep Research SOP Pipeline D baseline validation, which is a prerequisite for developing novel methods.

## Prerequisites
- Python 3.8+ with PyTorch/TensorFlow
- Docker (for reproducibility)
- Git and Git LFS
- Access to datasets (HuggingFace, academic repositories)

## What This Skill Does
1. **Extracts methodology** from papers and code repositories
2. **Validates datasets** match baseline specifications exactly
3. **Implements baseline** with exact hyperparameters
4. **Runs experiments** with deterministic settings
5. **Validates results** within ±1% statistical tolerance
6. **Creates reproducibility package** tested in fresh Docker environment
7. **Generates Quality Gate 1 validation** checklist

---

## Quick Start (30 minutes)

### Basic Replication
```bash
# 1. Specify baseline to replicate
BASELINE_PAPER="BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2019)"
BASELINE_CODE="https://github.com/google-research/bert"
TARGET_METRIC="Accuracy on SQuAD 2.0"
PUBLISHED_RESULT=0.948

# 2. Run replication workflow
./scripts/replicate-baseline.sh \
  --paper "$BASELINE_PAPER" \
  --code "$BASELINE_CODE" \
  --metric "$TARGET_METRIC" \
  --expected "$PUBLISHED_RESULT"

# 3. Review results
cat output/baseline-bert/replication-report.md
```

Expected output:
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

## Step-by-Step Guide

### Phase 1: Paper Analysis (15 minutes)

#### Extract Methodology
```bash
# Coordinate with researcher agent
./scripts/analyze-paper.sh --paper "arXiv:2103.00020"
```

The script extracts:
- Model architecture (layers, hidden sizes, attention heads)
- Training hyperparameters (learning rate, batch size, warmup)
- Optimization details (optimizer type, weight decay, dropout)
- Dataset specifications (splits, preprocessing, tokenization)
- Evaluation metrics (primary and secondary)

**Output**: `baseline-specification.md` with all extracted details

#### Validate Completeness
```bash
# Check for missing hyperparameters
./scripts/validate-spec.sh baseline-specification.md
```

**Common Missing Details**:
- Learning rate schedule (linear warmup, cosine decay)
- Random seeds (NumPy, PyTorch, Python)
- Hardware specifications (GPU type, memory)
- Framework versions (PyTorch 1.7 vs 1.13 numerical differences)

**If details missing**:
1. Check paper supplements
2. Check official code config files
3. Check GitHub issues
4. Contact authors

---

### Phase 2: Dataset Validation (20 minutes)

#### Coordinate with data-steward Agent
```bash
# Validate dataset matches baseline specs
./scripts/validate-dataset.sh \
  --dataset "SQuAD 2.0" \
  --splits "train:130k,dev:12k" \
  --preprocessing "WordPiece tokenization, max_length=384"
```

**data-steward checks**:
- Exact dataset version (v2.0, not v1.1)
- Sample counts match (training: 130,319 examples)
- Data splits match (80/10/10 vs 90/10)
- Preprocessing matches (lower-casing, accent stripping)
- Checksum validation (SHA256 hashes)

**Output**: `dataset-validation-report.md`

---

### Phase 3: Implementation (2 hours)

#### Coordinate with coder Agent
```bash
# Implement baseline with exact specifications
./scripts/implement-baseline.sh \
  --spec baseline-specification.md \
  --framework pytorch \
  --template resources/templates/bert-base.py
```

**coder creates**:
```python
# baseline-bert-implementation.py
import torch
import random
import numpy as np

# CRITICAL: Set all random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# CRITICAL: Enable deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Exact hyperparameters from paper
config = {
    "num_layers": 12,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "learning_rate": 5e-5,  # From paper section 4.2
    "batch_size": 32,       # Per-GPU batch size
    "num_epochs": 3,
    "warmup_steps": 10000,  # 10% of training steps
    "weight_decay": 0.01,
    "dropout": 0.1
}
```

**Unit Tests**:
```bash
pytest baseline-bert-implementation_test.py -v
```

**Output**: Fully tested implementation matching baseline exactly

---

### Phase 4: Experiment Execution (4-8 hours)

#### Coordinate with tester Agent
```bash
# Run experiments with monitoring
./scripts/run-experiments.sh \
  --implementation baseline-bert-implementation.py \
  --config config/bert-squad.yaml \
  --gpus 4 \
  --monitor true
```

**tester executes**:
1. **Environment Setup**:
   ```bash
   # Create deterministic environment
   docker build -t baseline-bert:v1.0 -f Dockerfile .
   docker run --gpus all -v $(pwd):/workspace baseline-bert:v1.0
   ```

2. **Training with Monitoring**:
   ```python
   # Log training curves
   from torch.utils.tensorboard import SummaryWriter
   writer = SummaryWriter('logs/baseline-bert')

   for epoch in range(3):
       for batch in dataloader:
           loss = model(batch)
           writer.add_scalar('Loss/train', loss, global_step)
           writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)
   ```

3. **Checkpoint Saving**:
   ```python
   # Save best checkpoint
   torch.save({
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'loss': loss,
       'accuracy': accuracy
   }, 'checkpoints/best-model.pt')
   ```

**Output**:
- `training.log` - Complete training logs
- `best-model.pt` - Best checkpoint
- `metrics.json` - All evaluation metrics

---

### Phase 5: Results Validation (30 minutes)

#### Statistical Comparison
```bash
# Compare reproduced vs published results
./scripts/compare-results.sh \
  --reproduced 0.945 \
  --published 0.948 \
  --tolerance 0.01
```

**Validation checks**:
```python
import scipy.stats as stats

# Paired t-test
reproduced = [0.945, 0.946, 0.944]  # 3 runs
published = 0.948

difference = np.mean(reproduced) - published
percent_diff = (difference / published) * 100

# Within tolerance?
within_tolerance = abs(difference / published) <= 0.01

# Statistical significance
t_stat, p_value = stats.ttest_1samp(reproduced, published)
confidence_interval = stats.t.interval(0.95, len(reproduced)-1,
                                       loc=np.mean(reproduced),
                                       scale=stats.sem(reproduced))

print(f"Reproduced: {np.mean(reproduced):.3f} ± {np.std(reproduced):.3f}")
print(f"Published: {published:.3f}")
print(f"Difference: {difference:.3f} ({percent_diff:.2f}%)")
print(f"Within ±1% tolerance: {within_tolerance}")
print(f"95% CI: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
```

**If results differ > 1%**:
```bash
# Debug systematically
./scripts/debug-divergence.sh \
  --reproduced 0.932 \
  --published 0.948
```

**Common causes**:
- Random seed not propagated to all libraries
- Framework version differences (PyTorch 1.7 vs 1.13)
- Hardware differences (V100 vs A100 numerical precision)
- Missing hyperparameter (learning rate schedule)
- Dataset preprocessing mismatch

**Output**: `baseline-bert-comparison.md` with statistical analysis

---

### Phase 6: Reproducibility Packaging (30 minutes)

#### Coordinate with archivist Agent
```bash
# Create complete reproducibility package
./scripts/create-repro-package.sh \
  --name baseline-bert \
  --code baseline-bert-implementation.py \
  --model best-model.pt \
  --env requirements.txt
```

**archivist creates**:
```
baseline-bert-repro.tar.gz
├── README.md                    # ≤5 steps to reproduce
├── requirements.txt             # Exact versions
├── Dockerfile                   # Exact environment
├── src/
│   ├── baseline-bert-implementation.py
│   ├── data_loader.py
│   └── train.py
├── data/
│   └── download_instructions.txt
├── models/
│   └── best-model.pt
├── logs/
│   └── training.log
├── results/
│   ├── metrics.json
│   └── comparison.csv
└── MANIFEST.txt                 # SHA256 checksums
```

**README.md (≤5 steps)**:
```markdown
# BERT SQuAD 2.0 Baseline Reproduction

## Quick Reproduction (3 steps)

1. Build Docker environment:
   ```bash
   docker build -t bert-squad:v1.0 .
   ```

2. Download SQuAD 2.0 dataset:
   ```bash
   ./download_data.sh
   ```

3. Run training:
   ```bash
   docker run --gpus all -v $(pwd):/workspace bert-squad:v1.0 python src/train.py
   ```

Expected result: 0.945 ± 0.001 accuracy (within ±1% of published 0.948)
```

#### Test Reproducibility
```bash
# Fresh Docker reproduction
./scripts/test-reproducibility.sh --package baseline-bert-repro.tar.gz --runs 3
```

**Output**: 3 successful reproductions with deterministic results

---

### Phase 7: Quality Gate 1 Validation (15 minutes)

#### Coordinate with evaluator Agent
```bash
# Validate Quality Gate 1 requirements
./scripts/validate-gate-1.sh --baseline baseline-bert
```

**evaluator checks**:
- ✅ Baseline specification complete (47/47 hyperparameters)
- ✅ Dataset validation passed
- ✅ Implementation tested (100% unit test coverage)
- ✅ Results within ±1% tolerance (0.945 vs 0.948)
- ✅ Reproducibility verified (3/3 fresh reproductions)
- ✅ Code documented and archived

**Decision Logic**:
```python
if results_within_tolerance and reproducibility_verified:
    decision = "APPROVED"
elif minor_gaps_fixable:
    decision = "CONDITIONAL"  # e.g., 1.2% difference but deterministic
else:
    decision = "REJECT"  # e.g., 5% difference, non-deterministic
```

**Output**: `gate-1-validation-checklist.md`

```markdown
# Quality Gate 1: Baseline Validation

## Status: APPROVED

### Requirements
- [x] Baseline specification document complete
- [x] Dataset validation passed
- [x] Implementation tested and reviewed
- [x] Results within ±1% of published (0.945 vs 0.948)
- [x] Reproducibility package tested in fresh environment
- [x] Documentation complete
- [x] All artifacts archived

### Evidence
- Baseline spec: `baseline-bert-specification.md`
- Dataset validation: `dataset-validation-report.md`
- Implementation: `baseline-bert-implementation.py` (100% test coverage)
- Results comparison: `baseline-bert-comparison.md`
- Reproducibility package: `baseline-bert-repro.tar.gz` (3/3 successful)

### Approval
**Date**: 2025-11-01
**Approved By**: evaluator agent
**Next Step**: Proceed to Pipeline D novel method development
```

---

## Advanced Features

### Multi-Baseline Comparison
```bash
# Compare multiple baselines simultaneously
./scripts/compare-baselines.sh \
  --baselines "bert-base,roberta-base,electra-base" \
  --dataset "SQuAD 2.0" \
  --metrics "accuracy,f1,em"
```

### Ablation Study Integration
```bash
# Once baseline validated, run ablations
./scripts/run-ablations.sh \
  --baseline baseline-bert \
  --ablations "no-warmup,no-weight-decay,smaller-lr"
```

### Continuous Validation
```bash
# Set up monitoring for baseline drift
./scripts/setup-monitoring.sh \
  --baseline baseline-bert \
  --schedule "weekly" \
  --alert-threshold 0.02
```

---

## Troubleshooting

### Issue: Missing Hyperparameters
**Symptoms**: Specification validation fails with missing details
**Cause**: Paper doesn't document all hyperparameters
**Solution**:
```bash
# Check official code config files
grep -r "learning_rate\|batch_size\|warmup" ${BASELINE_CODE}/

# Check GitHub issues
gh issue list --repo ${BASELINE_REPO} --search "hyperparameter"

# Contact authors
./scripts/contact-authors.sh --paper "arXiv:2103.00020" --question "learning rate schedule"
```

### Issue: Results Diverge > 1%
**Symptoms**: Reproduced 0.932, published 0.948 (1.7% difference)
**Solution**:
```bash
# Systematic debugging
./scripts/debug-divergence.sh --detailed

# Check random seeds
python -c "import torch; print(torch.initial_seed())"

# Check framework version
python -c "import torch; print(torch.__version__)"

# Enable detailed logging
python baseline-bert-implementation.py --debug --log-level DEBUG
```

### Issue: Non-Deterministic Results
**Symptoms**: 3 runs produce 0.945, 0.951, 0.938 (high variance)
**Solution**:
```python
# Force deterministic mode
import torch
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Check for non-deterministic operations
torch.set_deterministic(True)  # Will error if non-deterministic ops used
```

### Issue: Docker Environment Fails
**Symptoms**: Docker build or run errors
**Solution**:
```bash
# Check Docker resources
docker system df
docker system prune -a  # Free up space if needed

# Use pre-built base image
docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Debug interactively
docker run -it --gpus all pytorch/pytorch:1.13.1 bash
```

---

## Output Files

| File | Description | Size |
|------|-------------|------|
| `baseline-{method}-specification.md` | Extracted methodology | ~5KB |
| `dataset-validation-report.md` | Dataset validation results | ~2KB |
| `baseline-{method}-implementation.py` | Clean implementation | ~10KB |
| `baseline-{method}-implementation_test.py` | Unit tests | ~5KB |
| `training.log` | Complete training logs | ~100MB |
| `best-model.pt` | Best checkpoint | ~400MB |
| `metrics.json` | All evaluation metrics | ~1KB |
| `baseline-{method}-comparison.md` | Results comparison | ~3KB |
| `baseline-{method}-comparison.csv` | Metrics table | ~1KB |
| `baseline-{method}-repro.tar.gz` | Reproducibility package | ~450MB |
| `gate-1-validation-checklist.md` | Quality Gate 1 evidence | ~3KB |

---

## Integration with Deep Research SOP

### Pipeline D: Method Development
Baseline replication is **mandatory** before novel method development:

```
Pipeline D Flow:
1. Replicate Baseline (this skill) → Quality Gate 1
2. Develop Novel Method (method-development skill)
3. Ablation Studies (5+ ablations required)
4. Statistical Validation (p < 0.05)
5. Submit for Gate 2 review
```

### Agent Coordination

**Sequential workflow**:
```yaml
researcher:
  - Analyze paper
  - Extract methodology
  - Identify data sources
  ↓
data-steward:
  - Validate datasets
  - Check integrity
  - Verify preprocessing
  ↓
coder:
  - Implement baseline
  - Add unit tests
  - Code review
  ↓
tester:
  - Run experiments
  - Monitor training
  - Collect metrics
  ↓
archivist:
  - Create repro package
  - Test fresh reproduction
  - Archive artifacts
  ↓
evaluator:
  - Validate Gate 1
  - Generate checklist
  - Approve/Conditional/Reject
```

### Memory MCP Integration
```bash
# Store baseline specification
memory-store --key "sop/pipeline-d/baseline-bert/specification" \
             --value "$(cat baseline-bert-specification.md)" \
             --layer long_term

# Store validation results
memory-store --key "sop/pipeline-d/baseline-bert/gate-1" \
             --value "$(cat gate-1-validation-checklist.md)" \
             --layer long_term
```

---

## Related Skills

- **method-development** - Develop novel methods after baseline validation
- **holistic-evaluation** - Run HELM + CheckList evaluations (Pipeline E)
- **gate-validation** - Quality Gate approval workflow
- **reproducibility-audit** - Test reproducibility packages
- **literature-synthesis** - PRISMA systematic reviews

---

## Resources

### Official Standards
- [ACM Artifact Evaluation](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
- [ML Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
- [TIER Protocol](https://www.projecttier.org/)

### Tools
- [Docker](https://www.docker.com/) - Reproducible environments
- [DVC](https://dvc.org/) - Data version control
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [MLflow](https://mlflow.org/) - ML lifecycle management

### Deep Research SOP Documentation
- Pipeline D specification: `docs/deep-research-sop-gap-analysis.md`
- Quality Gates overview: `CLAUDE.md`
- Agent definitions: `agents/research/`
- Command specifications: `.claude/commands/research/`

---

**Created**: 2025-11-01
**Version**: 1.0.0
**Category**: Deep Research SOP
**Pipeline**: D (Method Development)
**Quality Gate**: 1 (Baseline Validation)
**Estimated Time**: 8-12 hours (first baseline), 4-6 hours (subsequent)