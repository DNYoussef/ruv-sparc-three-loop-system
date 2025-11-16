---
name: method-development
description: Develop novel machine learning methods with rigorous ablation studies
  for Deep Research SOP Pipeline D. Use after baseline replication passes Quality
  Gate 1, when creating new algorithms, proposing modifications to existing methods,
  or conducting systematic experimental validation. Includes architectural innovation,
  hyperparameter optimization, and component-wise ablation analysis leading to Quality
  Gate 2.
version: 1.0.0
category: research
tags:
- research
- analysis
- planning
author: ruv
---

# Method Development

Systematically develop and validate novel machine learning methods through controlled experimentation, ablation studies, and architectural innovation following Deep Research SOP Pipeline D.

## Overview

**Purpose**: Develop novel ML methods with rigorous experimental validation after baseline replication

**When to Use**:
- Quality Gate 1 (baseline replication) has APPROVED status
- Proposing architectural modifications to baseline methods
- Developing new training algorithms or optimization strategies
- Creating novel model components or attention mechanisms
- Systematic hyperparameter optimization required
- Ablation studies needed to validate design choices

**Quality Gate**: Leads to Quality Gate 2 (Model & Evaluation Validation)

**Prerequisites**:
- Baseline replication completed with ±1% tolerance (Quality Gate 1 passed)
- Baseline reproducibility package available
- Statistical analysis framework in place
- Docker environment configured
- GPU resources allocated (4-8 GPUs recommended)

**Outputs**:
- Novel method implementation with complete codebase
- Ablation study results (minimum 5 components tested)
- Performance comparison vs. baseline (statistical significance)
- Architectural diagrams and design documentation
- Hyperparameter sensitivity analysis
- Quality Gate 2 checklist (model validation requirements)

**Time Estimate**: 3-7 days (varies by complexity)
- Phase 1 (Architecture Design): 4-8 hours
- Phase 2 (Prototype Implementation): 1-2 days
- Phase 3 (Ablation Studies): 2-3 days
- Phase 4 (Optimization): 1-2 days
- Phase 5 (Comparative Evaluation): 4-8 hours
- Phase 6 (Documentation): 2-4 hours
- Phase 7 (Gate 2 Validation): 2-4 hours

**Agents Used**: system-architect, coder, tester, ethics-agent, reviewer, archivist, evaluator

---

## Quick Start

### 1. Prerequisites Check
```bash
# Verify baseline replication passed Gate 1
npx claude-flow@alpha memory retrieve --key "sop/gate-1/status"

# Load baseline reproducibility package
cd baseline-replication-package/
docker build -t baseline:latest .

# Verify baseline results
python scripts/verify_baseline_results.py --tolerance 0.01
```

### 2. Initialize Method Development
```bash
# Run architecture design workflow
npx claude-flow@alpha hooks pre-task \
  --description "Method development: Novel attention mechanism"

# Create method development workspace
mkdir -p novel-method/{src,experiments,ablations,docs}
cd novel-method/
```

### 3. Design Novel Architecture
```bash
# Invoke system-architect agent
# Document architectural decisions
# Create comparison diagrams (baseline vs. novel)
```

### 4. Run Ablation Studies
```bash
# Minimum 5 component ablations required
python scripts/run_ablations.py \
  --components "attention,normalization,residual,activation,pooling" \
  --baseline baseline:latest \
  --runs 3 \
  --seeds 42,123,456
```

### 5. Statistical Validation
```bash
# Compare novel method vs. baseline
python scripts/statistical_comparison.py \
  --method novel-method \
  --baseline baseline \
  --test paired-ttest \
  --significance 0.05
```

### 6. Quality Gate 2 Validation
```bash
# Validate Gate 2 requirements
npx claude-flow@alpha sparc run evaluator \
  "/validate-gate-2 --pipeline E --method novel-method"
```

---

## Detailed Instructions

### Phase 1: Architecture Design (4-8 hours)

**Agent**: system-architect

**Objectives**:
1. Analyze baseline architecture and identify improvement opportunities
2. Design novel components with theoretical justification
3. Create architectural diagrams showing modifications
4. Document design decisions and hypotheses

**Steps**:

#### 1.1 Baseline Architecture Analysis
```bash
# Load baseline architecture
python scripts/analyze_baseline_architecture.py \
  --checkpoint baseline-weights.pth \
  --output docs/baseline-architecture.md

# Identify bottlenecks
python scripts/profile_baseline.py \
  --mode training \
  --output profiling/baseline-profile.json
```

**Deliverable**: Baseline architecture analysis document

#### 1.2 Novel Architecture Design
Invoke system-architect agent with:
```
Design a novel architecture that improves upon the baseline by:
1. Addressing identified bottlenecks
2. Incorporating recent advances (Transformers, efficient attention, etc.)
3. Maintaining computational feasibility
4. Providing theoretical justification

Output:
- Architectural diagram (draw.io or GraphViz)
- Component specifications
- Computational complexity analysis (O(n) notation)
- Expected performance improvements with justification
```

**Deliverable**: Novel architecture specification document

#### 1.3 Hypothesis Formulation
Document testable hypotheses:
```markdown
## Hypotheses

### H1: Novel Attention Mechanism
**Claim**: Multi-scale attention improves long-range dependency modeling
**Baseline**: Standard scaled dot-product attention
**Expected**: +2-5% accuracy on sequence tasks
**Ablation**: Compare with/without multi-scale component

### H2: Residual Normalization
**Claim**: Pre-norm residual blocks stabilize training
**Baseline**: Post-norm residual blocks
**Expected**: 1.5x faster convergence
**Ablation**: Pre-norm vs. post-norm vs. no-norm

[Continue for each novel component...]
```

**Deliverable**: Hypotheses document with testable predictions

#### 1.4 Design Review
Coordinate with reviewer agent:
```bash
npx claude-flow@alpha hooks notify \
  --message "Architecture design complete, requesting review" \
  --recipients "reviewer"
```

**Review Checklist**:
- [ ] Theoretical justification sound
- [ ] Computational complexity acceptable
- [ ] Hypotheses testable with ablation studies
- [ ] Design maintains reproducibility (deterministic operations)
- [ ] Novel components well-documented

**Deliverable**: Approved architecture design

---

### Phase 2: Prototype Implementation (1-2 days)

**Agent**: coder

**Objectives**:
1. Implement novel architecture in PyTorch/TensorFlow
2. Maintain code quality (100% test coverage)
3. Enable deterministic mode for reproducibility
4. Create modular, ablation-ready codebase

**Steps**:

#### 2.1 Project Setup
```bash
# Initialize project structure
mkdir -p src/{models,layers,utils,config}
mkdir -p tests/{unit,integration,ablation}
mkdir -p experiments/{configs,scripts,results}

# Copy baseline code as starting point
cp -r ../baseline-replication-package/src/* src/

# Initialize Git repository with DVC
git init
dvc init
```

#### 2.2 Novel Component Implementation
Invoke coder agent with:
```
Implement the following novel components:

1. Multi-Scale Attention (src/layers/attention.py)
   - Support 3 scales: local, medium, global
   - Efficient implementation using sparse matrices
   - Deterministic mode with fixed seeds

2. Pre-Norm Residual Blocks (src/layers/residual.py)
   - Layer normalization before residual connection
   - Optional dropout for regularization

3. [Other novel components...]

Requirements:
- Type hints for all functions
- Docstrings with complexity analysis
- Unit tests achieving 100% coverage
- Ablation flags for each component
```

**Code Quality Standards**:
```python
# Example: Multi-scale attention with ablation support
class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism with local, medium, and global receptive fields.

    Computational Complexity:
    - Time: O(n * k) where n=sequence_length, k=num_scales
    - Space: O(n * d) where d=embedding_dimension

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads per scale
        num_scales: Number of scales (default: 3)
        ablate_scales: Disable specific scales for ablation (default: None)

    Example:
        >>> attn = MultiScaleAttention(embed_dim=512, num_heads=8)
        >>> # Ablation: disable global scale
        >>> attn_ablated = MultiScaleAttention(embed_dim=512, num_heads=8,
        ...                                     ablate_scales=['global'])
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_scales: int = 3,
        ablate_scales: Optional[List[str]] = None
    ):
        super().__init__()
        self.ablate_scales = ablate_scales or []
        # Implementation...
```

#### 2.3 Integration with Baseline
```bash
# Ensure backward compatibility
python tests/integration/test_baseline_equivalence.py \
  --novel-model src/models/novel_model.py \
  --baseline-model ../baseline-replication-package/src/models/baseline.py \
  --ablate-all-novel  # Should match baseline when all novel components disabled
```

**Deliverable**: Implemented novel method codebase

#### 2.4 Test Suite Development
Invoke tester agent with:
```
Create comprehensive test suite covering:

1. Unit Tests (tests/unit/)
   - Each novel component in isolation
   - Edge cases and boundary conditions
   - Numerical stability tests

2. Integration Tests (tests/integration/)
   - Novel model end-to-end training
   - Gradient flow validation
   - Memory profiling

3. Ablation Tests (tests/ablation/)
   - Each component can be disabled via flags
   - Ablated model runs without errors
   - Ablation results logged correctly

Target: 100% code coverage
```

**Deliverable**: Complete test suite with coverage report

---

### Phase 3: Ablation Studies (2-3 days)

**Agent**: tester

**Objectives**:
1. Systematically ablate each novel component
2. Measure performance impact with statistical significance
3. Identify critical vs. non-critical components
4. Validate architectural hypotheses

**Steps**:

#### 3.1 Ablation Configuration
Create ablation matrix in `experiments/ablation_matrix.yaml`:
```yaml
ablation_matrix:
  # Baseline: All novel components enabled
  - name: "baseline"
    ablations: []
    expected_metric: 0.850

  # Ablation 1: Disable multi-scale attention
  - name: "ablate_multiscale_attn"
    ablations: ["multiscale_attention"]
    hypothesis: "Should drop 2-5% accuracy"

  # Ablation 2: Disable pre-norm residual
  - name: "ablate_prenorm"
    ablations: ["prenorm_residual"]
    hypothesis: "Should converge 1.5x slower"

  # Ablation 3: Disable both (check interaction effects)
  - name: "ablate_attn_and_prenorm"
    ablations: ["multiscale_attention", "prenorm_residual"]
    hypothesis: "Should show super-additive degradation if components synergize"

  # [Continue for all 2^n combinations where n=number of novel components]
  # Minimum 5 components tested individually
```

#### 3.2 Run Ablation Experiments
```bash
# Run ablation suite with 3 seeds for statistical validity
python experiments/scripts/run_ablations.py \
  --config experiments/ablation_matrix.yaml \
  --seeds 42,123,456 \
  --gpus 4 \
  --output experiments/results/ablations/

# Expected runtime: 2-3 days depending on model complexity
```

**Monitoring**:
```bash
# Monitor progress in real-time
watch -n 60 'python scripts/check_ablation_progress.py'

# Alert on failures
python scripts/monitor_ablations.py \
  --alert-on-failure \
  --email your-email@example.com
```

#### 3.3 Statistical Analysis
```bash
# Analyze ablation results
python scripts/analyze_ablations.py \
  --results experiments/results/ablations/ \
  --baseline experiments/results/ablations/baseline/ \
  --test paired-ttest \
  --significance 0.05 \
  --bonferroni-correction \
  --output experiments/results/ablation_analysis.pdf
```

**Deliverable**: Ablation study report with statistical significance

#### 3.4 Component Importance Ranking
```python
# Generate component importance scores
from sklearn.ensemble import RandomForestRegressor

# Train meta-model: ablations -> performance
# Rank components by feature importance
python scripts/rank_component_importance.py \
  --ablations experiments/results/ablations/ \
  --method random-forest \
  --output docs/component_importance.md
```

**Deliverable**: Component importance ranking

---

### Phase 4: Hyperparameter Optimization (1-2 days)

**Agent**: coder

**Objectives**:
1. Optimize hyperparameters for novel method
2. Conduct sensitivity analysis
3. Document optimal configuration
4. Compare with baseline hyperparameters

**Steps**:

#### 4.1 Hyperparameter Search Space
```python
# Define search space in experiments/configs/hparam_search.yaml
search_space:
  learning_rate:
    type: log_uniform
    min: 1e-5
    max: 1e-2

  attention_heads:
    type: choice
    values: [4, 8, 16, 32]

  num_layers:
    type: int_uniform
    min: 6
    max: 24

  dropout:
    type: uniform
    min: 0.0
    max: 0.5

  # [Continue for all tunable hyperparameters]
```

#### 4.2 Bayesian Optimization
```bash
# Run Bayesian optimization with Optuna
python experiments/scripts/optimize_hyperparameters.py \
  --search-space experiments/configs/hparam_search.yaml \
  --n-trials 100 \
  --sampler TPE \
  --pruner MedianPruner \
  --output experiments/results/hparam_optimization/
```

#### 4.3 Sensitivity Analysis
```bash
# Analyze hyperparameter sensitivity
python scripts/sensitivity_analysis.py \
  --optimization-results experiments/results/hparam_optimization/ \
  --method sobol \
  --output docs/sensitivity_analysis.pdf
```

**Deliverable**: Optimal hyperparameter configuration

---

### Phase 5: Comparative Evaluation (4-8 hours)

**Agent**: tester

**Objectives**:
1. Compare novel method vs. baseline with statistical rigor
2. Evaluate on multiple metrics (accuracy, speed, memory)
3. Test generalization across datasets/splits
4. Document performance improvements

**Steps**:

#### 5.1 Benchmark Suite
```bash
# Run comprehensive benchmarks
python experiments/scripts/benchmark_comparison.py \
  --novel-method src/models/novel_model.py \
  --novel-checkpoint experiments/results/best_checkpoint.pth \
  --baseline ../baseline-replication-package/baseline.pth \
  --datasets "train,val,test" \
  --metrics "accuracy,f1,precision,recall,latency,memory" \
  --runs 5 \
  --output experiments/results/comparison/
```

#### 5.2 Statistical Comparison
```bash
# Paired t-test with Bonferroni correction
python scripts/statistical_comparison.py \
  --novel experiments/results/comparison/novel_*.json \
  --baseline experiments/results/comparison/baseline_*.json \
  --test paired-ttest \
  --correction bonferroni \
  --significance 0.05
```

**Expected Output**:
```
Novel Method vs. Baseline Comparison
====================================
Metric: Accuracy
  Novel:    0.875 ± 0.003
  Baseline: 0.850 ± 0.002
  Δ:        +2.5% (p=0.001, significant)

Metric: Latency (ms)
  Novel:    45.2 ± 1.1
  Baseline: 42.8 ± 0.9
  Δ:        +5.6% (p=0.01, significant)

Metric: Memory (GB)
  Novel:    8.4 ± 0.1
  Baseline: 7.2 ± 0.1
  Δ:        +16.7% (p<0.001, significant)

Conclusion: Novel method achieves significant accuracy improvement (+2.5%)
at the cost of increased latency (+5.6%) and memory (+16.7%).
Pareto-optimal for accuracy-critical applications.
```

**Deliverable**: Comparative evaluation report

#### 5.3 Generalization Testing
```bash
# Test on held-out datasets
python scripts/test_generalization.py \
  --model experiments/results/best_checkpoint.pth \
  --datasets "dataset2,dataset3,dataset4" \
  --output experiments/results/generalization/
```

**Deliverable**: Generalization analysis

---

### Phase 6: Documentation (2-4 hours)

**Agent**: archivist

**Objectives**:
1. Document novel method architecture
2. Create architectural diagrams
3. Write method card (similar to model card)
4. Prepare for Quality Gate 2

**Steps**:

#### 6.1 Method Card Creation
Coordinate with archivist agent:
```bash
npx claude-flow@alpha sparc run archivist \
  "Create method card for novel architecture following Mitchell et al. 2019 template"
```

**Method Card Sections**:
1. **Method Details**: Architecture, components, design rationale
2. **Intended Use**: Task types, domains, limitations
3. **Performance**: Metrics, comparisons, ablation results
4. **Training**: Hyperparameters, optimization, data requirements
5. **Computational Requirements**: GPU, memory, latency
6. **Ethical Considerations**: Bias, fairness, dual-use risks
7. **Caveats and Recommendations**: Known issues, best practices

#### 6.2 Architectural Diagrams
Create diagrams showing:
- High-level architecture comparison (baseline vs. novel)
- Novel component details (attention mechanism, residual blocks, etc.)
- Information flow diagrams
- Computational graph

**Tools**: draw.io, GraphViz, or LaTeX TikZ

#### 6.3 Reproducibility Documentation
```markdown
# Reproducibility Guide

## Environment Setup
\`\`\`bash
# Docker image
docker pull novel-method:v1.0

# Or build from source
docker build -t novel-method:v1.0 -f Dockerfile .
\`\`\`

## Training from Scratch
\`\`\`bash
python train.py \
  --config experiments/configs/optimal_hparams.yaml \
  --seed 42 \
  --deterministic \
  --output experiments/results/reproduction/
\`\`\`

## Expected Results
- Test Accuracy: 0.875 ± 0.003
- Training Time: ~48 hours on 4x V100 GPUs
- Final Checkpoint: experiments/results/reproduction/checkpoint_epoch_100.pth
\`\`\`
```

**Deliverable**: Complete documentation package

---

### Phase 7: Quality Gate 2 Validation (2-4 hours)

**Agent**: evaluator

**Objectives**:
1. Validate all Gate 2 requirements
2. Coordinate ethics review with ethics-agent
3. Generate Gate 2 checklist
4. Obtain APPROVED, CONDITIONAL, or REJECTED status

**Steps**:

#### 7.1 Gate 2 Requirements Check
Run evaluator agent:
```bash
npx claude-flow@alpha sparc run evaluator \
  "/validate-gate-2 --pipeline E --method novel-method --include-ethics"
```

**Gate 2 Requirements** (from Deep Research SOP):
- [ ] Novel method implemented with full codebase
- [ ] Ablation studies completed (minimum 5 components)
- [ ] Statistical comparison vs. baseline (p < 0.05 for improvements)
- [ ] Method card completed (≥90% sections filled)
- [ ] Ethics review APPROVED (from ethics-agent)
- [ ] Reproducibility tested (3/3 successful runs)
- [ ] Performance meets or exceeds baseline
- [ ] Code quality: 100% test coverage, linting passed
- [ ] Documentation: README with ≤5 steps to reproduce

#### 7.2 Ethics Review Coordination
Coordinate with ethics-agent:
```bash
npx claude-flow@alpha sparc run ethics-agent \
  "/assess-risks --component model --gate 2"
```

**Ethics Review Domains** (Gate 2):
1. Safety Risks: Harmful outputs, adversarial robustness
2. Fairness Risks: Model bias, demographic parity
3. Privacy Risks: Data leakage, membership inference

**Deliverable**: Ethics review approval

#### 7.3 Gate 2 Decision
Based on evaluator agent's assessment:

**APPROVED**: All critical requirements met, proceed to holistic evaluation
**CONDITIONAL**: Minor gaps, mitigations in progress, proceed with restrictions
**REJECTED**: Unmitigated critical issues, return to method development

#### 7.4 Memory Storage
Store Gate 2 results:
```bash
npx claude-flow@alpha memory store \
  --key "sop/gate-2/status" \
  --value "APPROVED" \
  --metadata '{"method": "novel-method", "accuracy": 0.875, "date": "2025-11-01"}'
```

**Deliverable**: Gate 2 checklist and decision

---

## Integration with Deep Research SOP

### Pipeline Integration
- **Pipeline D (Method Development)**: This skill implements the complete method development phase
- **Prerequisite**: Baseline replication (Quality Gate 1 APPROVED)
- **Next Step**: Holistic evaluation (Quality Gate 2 APPROVED required)

### Quality Gates
- **Gate 1**: Must pass before invoking this skill
- **Gate 2**: Validation performed in Phase 7 of this skill
- **Gate 3**: Archival and deployment (requires Gate 2 APPROVED)

### Agent Coordination
```
Flow: system-architect → coder → tester → reviewer → ethics-agent → archivist → evaluator

Phase 1: system-architect designs novel architecture
Phase 2: coder implements with 100% test coverage
Phase 3: tester runs ablation studies
Phase 4: coder optimizes hyperparameters
Phase 5: tester performs comparative evaluation
Phase 6: archivist creates documentation
Phase 7: evaluator validates Gate 2 + ethics-agent reviews safety/ethics
```

### Memory Coordination
All agents store/retrieve via Memory MCP:
```bash
# Store architectural decisions
npx claude-flow@alpha memory store \
  --key "sop/method-development/architecture" \
  --value "$(cat docs/architecture.md)"

# Retrieve baseline results for comparison
npx claude-flow@alpha memory retrieve \
  --key "sop/baseline-replication/results"
```

---

## Troubleshooting

### Issue: Novel method underperforms baseline
**Symptoms**: Novel method achieves lower accuracy than baseline
**Diagnosis**:
1. Check ablation study results - which components hurt performance?
2. Verify hyperparameter optimization converged
3. Test on validation set (not just training set)

**Solutions**:
```bash
# Re-run ablation studies with finer granularity
python experiments/scripts/run_ablations.py --fine-grained

# Extend hyperparameter search
python experiments/scripts/optimize_hyperparameters.py --n-trials 500

# Check for implementation bugs
python tests/integration/test_gradient_flow.py
python tests/integration/test_numerical_stability.py
```

### Issue: Ablation studies show no significant differences
**Symptoms**: All ablations yield similar performance (p > 0.05)
**Diagnosis**: Novel components may not be contributing meaningfully
**Solutions**:
1. Increase ablation granularity (test sub-components)
2. Verify components are actually being used (check forward pass)
3. Increase number of runs for statistical power
4. Consider larger effect sizes (stronger architectural changes)

### Issue: Gate 2 validation rejected
**Symptoms**: evaluator agent returns REJECTED status
**Common Causes**:
- Ethics review flagged critical risks
- Reproducibility tests failed (non-deterministic behavior)
- Performance regression vs. baseline
- Incomplete documentation

**Solutions**:
```bash
# Check Gate 2 requirements
npx claude-flow@alpha sparc run evaluator \
  "/validate-gate-2 --pipeline E --method novel-method --verbose"

# Address ethics concerns
npx claude-flow@alpha sparc run ethics-agent \
  "/assess-risks --component model --gate 2 --mitigation-plan"

# Fix reproducibility
python scripts/test_determinism.py --runs 10 --strict
```

### Issue: Computational resource exhaustion
**Symptoms**: OOM errors, extremely slow training
**Solutions**:
```bash
# Enable gradient checkpointing
python train.py --gradient-checkpointing

# Reduce batch size
python train.py --batch-size 16  # Instead of 32

# Use mixed precision training
python train.py --fp16

# Profile memory usage
python scripts/profile_memory.py --model novel-method
```

---

## Related Skills and Commands

### Prerequisites
- `baseline-replication` - Must complete before invoking this skill

### Next Steps (after Gate 2 APPROVED)
- `holistic-evaluation` - Comprehensive model evaluation across multiple dimensions
- `reproducibility-audit` - Audit reproducibility package before archival

### Related Commands
- `/validate-gate-2` - Gate 2 validation (evaluator agent)
- `/assess-risks` - Ethics review for models (ethics-agent)
- `/init-model-card` - Create model card (archivist agent)

### Parallel Skills
- `literature-synthesis` - Can run in parallel to gather SOTA comparisons

---

## References

### Academic Standards
- Mitchell et al. (2019): Model Cards for Model Reporting
- Sculley et al. (2015): Hidden Technical Debt in Machine Learning Systems
- Lipton & Steinhardt (2019): Troubling Trends in Machine Learning Scholarship

### Reproducibility Standards
- NeurIPS Reproducibility Checklist
- ACM Artifact Evaluation Badging
- Papers with Code Reproducibility Guidelines

### Statistical Methods
- Dror et al. (2018): The Hitchhiker's Guide to Testing Statistical Significance in NLP
- Bonferroni Correction for Multiple Comparisons
- Paired T-Tests for Method Comparison

---

## Appendix

### Example Ablation Study Results

```
Ablation Study: Multi-Scale Attention Mechanism
================================================

Configuration: ResNet-50 + Multi-Scale Attention on ImageNet

| Ablation                  | Accuracy | Δ from Full | p-value | Significant |
|---------------------------|----------|-------------|---------|-------------|
| Full Model                | 0.875    | -           | -       | -           |
| Ablate Local Scale        | 0.868    | -0.7%       | 0.032   | Yes         |
| Ablate Medium Scale       | 0.871    | -0.4%       | 0.156   | No          |
| Ablate Global Scale       | 0.852    | -2.3%       | 0.001   | Yes         |
| Ablate All (Baseline)     | 0.850    | -2.5%       | <0.001  | Yes         |

Conclusion: Global scale is critical (+2.3% over baseline), local scale contributes moderately (+0.7%), medium scale is non-significant.

Recommendation: Keep global and local scales, consider removing medium scale to reduce computational cost.
```

### Example Method Card Template

```markdown
# Method Card: Multi-Scale Attention ResNet

## Method Details
**Architecture**: ResNet-50 with Multi-Scale Attention
**Novel Components**:
- Multi-Scale Attention (local, medium, global)
- Pre-Norm Residual Blocks
**Design Rationale**: Improve long-range dependency modeling while maintaining computational efficiency

## Intended Use
**Tasks**: Image classification, object detection
**Domains**: Computer vision, medical imaging
**Limitations**: Requires GPU with ≥16GB memory

## Performance
**ImageNet Accuracy**: 87.5% (±0.3%)
**Baseline Comparison**: +2.5% over ResNet-50
**Latency**: 45.2ms per image (batch=32)

## Training
**Hyperparameters**: lr=1e-4, batch=256, epochs=100
**Optimizer**: AdamW with cosine annealing
**Data**: ImageNet-1k (1.28M images)

## Computational Requirements
**GPUs**: 4x V100 (16GB each)
**Training Time**: 48 hours
**Memory**: 8.4GB per GPU

## Ethical Considerations
**Bias**: Evaluated on Balanced Faces dataset, demographic parity within 2%
**Fairness**: No disparate impact detected (p > 0.05)
**Dual-Use**: Standard image classification, low dual-use risk

## Caveats
- Requires deterministic mode for reproducibility (may impact performance by ~1%)
- Not tested on extremely high-resolution images (>2048px)
- Best performance with batch size ≥32
```