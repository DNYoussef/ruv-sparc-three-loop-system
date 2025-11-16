---
name: holistic-evaluation
description: Comprehensive multi-dimensional model evaluation across accuracy, fairness,
  robustness, efficiency, interpretability, and safety for Deep Research SOP Pipeline
  E. Use after method development when Quality Gate 2 validation is required, ensuring
  models meet production-ready standards across 6+ evaluation dimensions before deployment.
version: 1.0.0
category: quality
tags:
- quality
- testing
- validation
author: ruv
---

# Holistic Evaluation

Systematically evaluate machine learning models across 6+ critical dimensions following Deep Research SOP Pipeline E, ensuring comprehensive assessment beyond simple accuracy metrics.

## Overview

**Purpose**: Comprehensive model evaluation across multiple dimensions for production readiness

**When to Use**:
- Method development complete (novel method implemented)
- Quality Gate 2 validation required
- Before model deployment to production
- Regulatory compliance evaluation needed (EU AI Act, FDA)
- Fairness, safety, and robustness assessment required
- Model comparison across multiple dimensions

**Quality Gate**: Required for Quality Gate 2 APPROVED status

**Prerequisites**:
- Trained model checkpoint available
- Method development completed
- Test datasets prepared (standard + adversarial + fairness)
- Evaluation framework installed (sklearn, fairness-indicators, etc.)
- Ethics review initiated

**Outputs**:
- Holistic evaluation report across 6+ dimensions
- Fairness metrics (demographic parity, equalized odds, etc.)
- Robustness analysis (adversarial, distribution shift)
- Efficiency metrics (latency, throughput, memory, energy)
- Interpretability analysis (SHAP, attention visualizations)
- Safety evaluation (harmful outputs, bias, privacy)
- Quality Gate 2 validation checklist

**Time Estimate**: 2-5 days
- Phase 1 (Accuracy Evaluation): 4-8 hours
- Phase 2 (Fairness Evaluation): 1 day
- Phase 3 (Robustness Testing): 1-2 days
- Phase 4 (Efficiency Profiling): 4-8 hours
- Phase 5 (Interpretability Analysis): 4-8 hours
- Phase 6 (Safety Evaluation): 1 day
- Phase 7 (Synthesis & Gate 2): 2-4 hours

**Agents Used**: tester, ethics-agent, archivist, evaluator

---

## Quick Start

### 1. Prerequisites Check
```bash
# Verify method development complete
npx claude-flow@alpha memory retrieve --key "sop/method-development/status"

# Load model checkpoint
python scripts/load_checkpoint.py \
  --checkpoint experiments/results/best_checkpoint.pth \
  --verify
```

### 2. Initialize Evaluation Framework
```bash
# Install evaluation libraries
pip install fairness-indicators foolbox interpret-ml carbontracker

# Download evaluation datasets
python scripts/download_eval_datasets.py \
  --datasets "standard,fairness,adversarial,ood"
```

### 3. Run Holistic Evaluation Suite
```bash
# Comprehensive evaluation across all dimensions
python scripts/holistic_evaluation.py \
  --model experiments/results/best_checkpoint.pth \
  --dimensions "accuracy,fairness,robustness,efficiency,interpretability,safety" \
  --output experiments/results/holistic_evaluation/
```

### 4. Generate Evaluation Report
```bash
# Synthesize results into comprehensive report
python scripts/generate_evaluation_report.py \
  --results experiments/results/holistic_evaluation/ \
  --template templates/holistic_evaluation_template.md \
  --output docs/holistic_evaluation_report.pdf
```

### 5. Quality Gate 2 Validation
```bash
# Validate Gate 2 with holistic evaluation results
npx claude-flow@alpha sparc run evaluator \
  "/validate-gate-2 --pipeline E --evaluation-report docs/holistic_evaluation_report.pdf"
```

---

## Detailed Instructions

### Phase 1: Accuracy Evaluation (4-8 hours)

**Agent**: tester

**Objectives**:
1. Evaluate standard metrics across test sets
2. Measure performance on challenging subsets
3. Compare with baselines and SOTA methods
4. Statistical significance testing

**Steps**:

#### 1.1 Standard Metrics Evaluation
```bash
# Evaluate on standard test set
python scripts/evaluate_accuracy.py \
  --model experiments/results/best_checkpoint.pth \
  --dataset test \
  --metrics "accuracy,precision,recall,f1,auc" \
  --output experiments/results/holistic_evaluation/accuracy/
```

**Expected Metrics** (classification example):
- Accuracy: Overall correctness
- Precision: Positive prediction quality
- Recall: Positive class coverage
- F1 Score: Harmonic mean of precision/recall
- AUC-ROC: Classifier discrimination ability

#### 1.2 Per-Class Performance
```bash
# Analyze per-class metrics
python scripts/per_class_analysis.py \
  --model experiments/results/best_checkpoint.pth \
  --dataset test \
  --output experiments/results/holistic_evaluation/accuracy/per_class.json
```

**Identify**:
- Worst-performing classes (bottom 10%)
- Confusion matrix patterns
- Class imbalance effects

#### 1.3 Error Analysis
```bash
# Systematic error analysis
python scripts/error_analysis.py \
  --predictions experiments/results/holistic_evaluation/accuracy/predictions.json \
  --dataset test \
  --error-types "false_positives,false_negatives,high_confidence_errors" \
  --output experiments/results/holistic_evaluation/accuracy/error_analysis/
```

**Deliverable**: Accuracy evaluation report with error analysis

---

### Phase 2: Fairness Evaluation (1 day)

**Agent**: ethics-agent

**Objectives**:
1. Measure fairness across demographic groups
2. Detect and quantify bias
3. Assess disparate impact
4. Recommend mitigation strategies

**Steps**:

#### 2.1 Demographic Parity Analysis
```python
# Evaluate demographic parity across sensitive attributes
from fairness_indicators import FairnessIndicators

fi = FairnessIndicators(
    model=model,
    dataset=test_dataset,
    sensitive_attributes=['race', 'gender', 'age']
)

# Demographic parity: P(Y_hat=1 | A=a) should be equal across groups
demographic_parity = fi.demographic_parity()
print(f"Demographic Parity Difference: {demographic_parity['max_difference']}")
# Target: < 0.10 (10% difference)
```

#### 2.2 Equalized Odds
```python
# Equalized odds: TPR and FPR should be equal across groups
equalized_odds = fi.equalized_odds()
print(f"Equalized Odds Difference: {equalized_odds['tpr_diff']}, {equalized_odds['fpr_diff']}")
# Target: < 0.10 for both TPR and FPR
```

#### 2.3 Calibration Across Groups
```python
# Calibration: P(Y=1 | Y_hat=p) should match predicted probability p across groups
calibration = fi.calibration_by_group()
fi.plot_calibration_curves()
```

#### 2.4 Intersectional Fairness
```bash
# Analyze intersectional groups (e.g., race × gender)
python scripts/intersectional_fairness.py \
  --model experiments/results/best_checkpoint.pth \
  --attributes "race,gender,age" \
  --intersections "race×gender,race×age" \
  --output experiments/results/holistic_evaluation/fairness/
```

#### 2.5 Bias Mitigation Recommendations
Coordinate with ethics-agent:
```bash
npx claude-flow@alpha sparc run ethics-agent \
  "/assess-risks --component model --gate 2 --focus fairness" \
  --fairness-results experiments/results/holistic_evaluation/fairness/
```

**Deliverable**: Fairness evaluation report with bias mitigation plan

---

### Phase 3: Robustness Testing (1-2 days)

**Agent**: tester

**Objectives**:
1. Adversarial robustness evaluation
2. Out-of-distribution (OOD) detection
3. Distribution shift resilience
4. Uncertainty calibration

**Steps**:

#### 3.1 Adversarial Robustness (White-Box)
```python
# FGSM, PGD, C&W attacks
from foolbox import PyTorchModel
from foolbox.attacks import FGSM, PGD, CarliniWagnerL2Attack

fmodel = PyTorchModel(model, bounds=(0, 1))

# Fast Gradient Sign Method (FGSM)
attack = FGSM()
adversarial_examples = attack(fmodel, images, labels, epsilons=[0.01, 0.03, 0.05])

# Projected Gradient Descent (PGD)
attack = PGD()
adversarial_examples = attack(fmodel, images, labels, epsilons=[0.01, 0.03, 0.05])

# Report robust accuracy
python scripts/adversarial_eval.py \
  --model experiments/results/best_checkpoint.pth \
  --attacks "fgsm,pgd,cw" \
  --epsilons "0.01,0.03,0.05,0.1" \
  --output experiments/results/holistic_evaluation/robustness/adversarial/
```

**Expected Results**:
```
Adversarial Robustness Results
==============================
Clean Accuracy: 87.5%

FGSM (ε=0.03): 62.3% (-25.2%)
PGD (ε=0.03):  45.8% (-41.7%)
C&W (ε=0.03):  38.1% (-49.4%)

Conclusion: Model vulnerable to white-box adversarial attacks.
Recommendation: Consider adversarial training.
```

#### 3.2 Out-of-Distribution (OOD) Detection
```bash
# Test on OOD datasets (e.g., CIFAR-10 trained on ImageNet tested)
python scripts/ood_detection.py \
  --model experiments/results/best_checkpoint.pth \
  --in-distribution ImageNet \
  --out-distributions "Places365,iNaturalist,Textures" \
  --metrics "auroc,fpr_at_tpr95" \
  --output experiments/results/holistic_evaluation/robustness/ood/
```

**Metrics**:
- AUROC: Area under ROC curve (higher = better OOD detection)
- FPR@95%TPR: False positive rate when TPR=95% (lower = better)

#### 3.3 Distribution Shift Resilience
```bash
# Test on corrupted data (Gaussian noise, blur, weather effects)
python scripts/distribution_shift_eval.py \
  --model experiments/results/best_checkpoint.pth \
  --corruptions "gaussian_noise,shot_noise,motion_blur,fog,snow" \
  --severities 1,2,3,4,5 \
  --output experiments/results/holistic_evaluation/robustness/corruption/
```

#### 3.4 Uncertainty Calibration
```python
# Expected Calibration Error (ECE)
from netcal.metrics import ECE

ece = ECE(bins=15)
calibration_error = ece.measure(confidences, predictions, ground_truth)
print(f"Expected Calibration Error: {calibration_error:.4f}")
# Target: ECE < 0.05

# Reliability diagrams
python scripts/plot_calibration.py \
  --predictions experiments/results/holistic_evaluation/robustness/predictions.json \
  --output experiments/results/holistic_evaluation/robustness/calibration.pdf
```

**Deliverable**: Robustness evaluation report

---

### Phase 4: Efficiency Profiling (4-8 hours)

**Agent**: tester

**Objectives**:
1. Latency measurement (inference time)
2. Throughput analysis (queries per second)
3. Memory profiling (RAM, VRAM)
4. Energy consumption estimation
5. Model size and compression

**Steps**:

#### 4.1 Latency Profiling
```bash
# Measure inference latency across batch sizes
python scripts/profile_latency.py \
  --model experiments/results/best_checkpoint.pth \
  --batch-sizes 1,8,16,32,64 \
  --iterations 1000 \
  --device cuda \
  --output experiments/results/holistic_evaluation/efficiency/latency.json
```

**Expected Output**:
```
Latency Profiling Results
=========================
Batch Size |  Mean (ms) | Std (ms) | P50 | P95 | P99
-----------|------------|----------|-----|-----|-----
1          |  12.3      | 0.5      | 12.2| 13.1| 13.8
8          |  45.2      | 1.2      | 45.0| 47.3| 48.9
16         |  78.5      | 2.1      | 78.1| 81.8| 84.2
32         |  142.7     | 3.5      | 142.0| 148.5| 152.3
```

#### 4.2 Throughput Measurement
```bash
# Queries per second (QPS)
python scripts/measure_throughput.py \
  --model experiments/results/best_checkpoint.pth \
  --duration 60 \
  --batch-size 32 \
  --device cuda \
  --output experiments/results/holistic_evaluation/efficiency/throughput.json
```

#### 4.3 Memory Profiling
```python
# GPU memory profiling
import torch

torch.cuda.reset_peak_memory_stats()
model.eval()
with torch.no_grad():
    output = model(input_batch)
peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
print(f"Peak GPU Memory: {peak_memory:.2f} GB")

# CPU memory profiling
from memory_profiler import profile
@profile
def inference():
    model(input_batch)
```

#### 4.4 Energy Consumption
```bash
# Estimate CO2 emissions and energy usage
pip install carbontracker

python scripts/energy_profiling.py \
  --model experiments/results/best_checkpoint.pth \
  --iterations 1000 \
  --output experiments/results/holistic_evaluation/efficiency/energy.json
```

#### 4.5 Model Compression Analysis
```bash
# Compare full vs. quantized vs. pruned models
python scripts/compression_comparison.py \
  --full-model experiments/results/best_checkpoint.pth \
  --quantized experiments/results/quantized_model.pth \
  --pruned experiments/results/pruned_model.pth \
  --metrics "size,latency,accuracy" \
  --output experiments/results/holistic_evaluation/efficiency/compression.json
```

**Deliverable**: Efficiency profiling report

---

### Phase 5: Interpretability Analysis (4-8 hours)

**Agent**: tester

**Objectives**:
1. Feature importance analysis (SHAP, LIME)
2. Attention visualization (for Transformer models)
3. Saliency maps (for vision models)
4. Counterfactual explanations
5. Model transparency documentation

**Steps**:

#### 5.1 SHAP (SHapley Additive exPlanations)
```python
import shap

# Initialize SHAP explainer
explainer = shap.DeepExplainer(model, background_data)

# Compute SHAP values
shap_values = explainer.shap_values(test_data)

# Visualize feature importance
shap.summary_plot(shap_values, test_data, plot_type="bar")
shap.summary_plot(shap_values, test_data)  # Beeswarm plot

# Save plots
plt.savefig("experiments/results/holistic_evaluation/interpretability/shap_summary.pdf")
```

#### 5.2 Attention Visualization (Transformers)
```python
# Extract and visualize attention weights
from bertviz import head_view, model_view

attention = model(input_ids, output_attentions=True).attentions
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Head view: attention patterns for each head
head_view(attention, tokens)

# Model view: attention across all layers
model_view(attention, tokens)
```

#### 5.3 Saliency Maps (Vision Models)
```python
from captum.attr import IntegratedGradients, Saliency

# Integrated Gradients
ig = IntegratedGradients(model)
attributions = ig.attribute(input_image, target=predicted_class)

# Saliency
saliency = Saliency(model)
grads = saliency.attribute(input_image, target=predicted_class)

# Visualize
from captum.attr import visualization as viz
viz.visualize_image_attr_multiple(
    attributions,
    original_image,
    methods=["original_image", "heat_map"],
    signs=["all", "absolute_value"],
    titles=["Original", "Attribution Magnitude"]
)
```

#### 5.4 Counterfactual Explanations
```bash
# Generate counterfactual examples
python scripts/generate_counterfactuals.py \
  --model experiments/results/best_checkpoint.pth \
  --dataset test \
  --method "diverse_counterfactual_explanations" \
  --num-examples 100 \
  --output experiments/results/holistic_evaluation/interpretability/counterfactuals/
```

**Deliverable**: Interpretability analysis report

---

### Phase 6: Safety Evaluation (1 day)

**Agent**: ethics-agent

**Objectives**:
1. Harmful output detection
2. Bias amplification analysis
3. Privacy leakage testing (membership inference)
4. Adversarial prompt testing (for LLMs)
5. Dual-use risk assessment

**Steps**:

#### 6.1 Harmful Output Detection
Coordinate with ethics-agent:
```bash
npx claude-flow@alpha sparc run ethics-agent \
  "/safety-eval --model experiments/results/best_checkpoint.pth --adversarial-testing"
```

**Test Categories**:
- Toxic language generation
- Hate speech propagation
- Misinformation amplification
- Harmful stereotypes

#### 6.2 Bias Amplification
```python
# Test if model amplifies biases present in training data
python scripts/bias_amplification_test.py \
  --model experiments/results/best_checkpoint.pth \
  --baseline-bias 0.15  # Bias level in training data \
  --output experiments/results/holistic_evaluation/safety/bias_amplification.json

# Expected: Model bias ≤ Baseline bias (no amplification)
```

#### 6.3 Privacy Leakage (Membership Inference)
```bash
# Membership inference attack
python scripts/membership_inference_attack.py \
  --model experiments/results/best_checkpoint.pth \
  --train-data train_dataset \
  --test-data test_dataset \
  --output experiments/results/holistic_evaluation/safety/privacy.json
```

**Metrics**:
- Attack Accuracy: Should be ≈50% (random guess) for privacy-preserving models
- Attack AUC: Should be ≈0.5

#### 6.4 Adversarial Prompt Testing (LLMs)
```bash
# Test with adversarial prompts (jailbreak attempts)
python scripts/adversarial_prompt_testing.py \
  --model experiments/results/best_checkpoint.pth \
  --prompt-categories "jailbreak,prompt_injection,context_manipulation" \
  --output experiments/results/holistic_evaluation/safety/adversarial_prompts.json
```

#### 6.5 Dual-Use Risk Assessment
Coordinate with ethics-agent:
```bash
npx claude-flow@alpha sparc run ethics-agent \
  "/assess-risks --component deployment --gate 3 --focus dual-use"
```

**Deliverable**: Safety evaluation report

---

### Phase 7: Synthesis & Gate 2 Validation (2-4 hours)

**Agent**: evaluator

**Objectives**:
1. Synthesize all evaluation dimensions
2. Generate comprehensive holistic evaluation report
3. Identify strengths, weaknesses, trade-offs
4. Validate Quality Gate 2 requirements
5. Provide deployment recommendations

**Steps**:

#### 7.1 Results Synthesis
```bash
# Aggregate all evaluation results
python scripts/synthesize_holistic_evaluation.py \
  --accuracy experiments/results/holistic_evaluation/accuracy/ \
  --fairness experiments/results/holistic_evaluation/fairness/ \
  --robustness experiments/results/holistic_evaluation/robustness/ \
  --efficiency experiments/results/holistic_evaluation/efficiency/ \
  --interpretability experiments/results/holistic_evaluation/interpretability/ \
  --safety experiments/results/holistic_evaluation/safety/ \
  --output docs/holistic_evaluation_report.md
```

#### 7.2 Holistic Evaluation Report Template
```markdown
# Holistic Evaluation Report: [Model Name]

## Executive Summary
- **Overall Assessment**: [APPROVED / CONDITIONAL / REJECTED]
- **Strengths**: High accuracy (87.5%), good calibration (ECE=0.03)
- **Weaknesses**: Vulnerable to adversarial attacks, fairness gaps in [group]
- **Trade-offs**: +2.5% accuracy vs. +5.6% latency vs. baseline
- **Deployment Recommendation**: [Production-ready / Requires mitigation / Not ready]

## 1. Accuracy Evaluation
- **Test Accuracy**: 87.5% (±0.3%)
- **Per-Class Performance**: [details]
- **Error Analysis**: [top error categories]
- **Status**: ✅ PASS (≥85% target)

## 2. Fairness Evaluation
- **Demographic Parity**: Max difference = 8.2% (target: <10%)
- **Equalized Odds (TPR)**: 6.5% difference (target: <10%)
- **Equalized Odds (FPR)**: 4.3% difference (target: <10%)
- **Intersectional Fairness**: [worst group] at 79.2% accuracy
- **Status**: ✅ PASS with minor gaps

## 3. Robustness Testing
- **Adversarial Robustness (PGD, ε=0.03)**: 45.8% (-41.7%)
- **OOD Detection (AUROC)**: 0.82 (target: >0.80)
- **Corruption Robustness (mCE)**: 68.3 (target: <75.0)
- **Calibration (ECE)**: 0.03 (target: <0.05)
- **Status**: ⚠️ CONDITIONAL (adversarial vulnerability)

## 4. Efficiency Profiling
- **Latency (batch=32)**: 142.7ms (target: <200ms)
- **Throughput**: 224 QPS (target: >100 QPS)
- **Memory**: 8.4GB (target: <16GB)
- **Energy (1000 inferences)**: 0.15 kWh, 0.06 kg CO2eq
- **Status**: ✅ PASS

## 5. Interpretability Analysis
- **SHAP Feature Importance**: Top 3 features identified
- **Attention Visualization**: [key patterns]
- **Saliency Maps**: Focuses on relevant regions
- **Counterfactuals**: Minimal changes required for class flip
- **Status**: ✅ PASS

## 6. Safety Evaluation
- **Harmful Output Rate**: 0.02% (target: <0.05%)
- **Bias Amplification**: No amplification detected
- **Privacy (Membership Inference AUC)**: 0.52 (target: ≤0.55)
- **Adversarial Prompts**: 95% rejection rate (target: >90%)
- **Dual-Use Risk**: Low (ethics-agent assessment)
- **Status**: ✅ PASS

## 7. Overall Assessment

### Strengths
1. High accuracy (87.5%) with good calibration
2. Fair across demographic groups (demographic parity <10%)
3. Efficient (142ms latency, 224 QPS)
4. Good interpretability (SHAP, attention visualization)
5. Safe (low harmful output rate, privacy-preserving)

### Weaknesses
1. Vulnerable to white-box adversarial attacks (PGD: 45.8%)
2. Fairness gaps in intersectional groups ([group]: 79.2%)
3. OOD detection could be improved (AUROC=0.82)

### Trade-offs
- **Accuracy vs. Robustness**: High clean accuracy (87.5%) but lower adversarial robustness (45.8%)
- **Efficiency vs. Accuracy**: +2.5% accuracy vs. +5.6% latency vs. baseline
- **Fairness vs. Accuracy**: Overall fairness acceptable, but 8.2% gap in some groups

### Deployment Recommendations

**Production-Ready Scenarios**:
- Standard inference workloads (non-adversarial)
- Fairness-critical applications with minor mitigations
- Latency-tolerant systems (<200ms acceptable)

**Mitigation Required**:
1. **Adversarial Robustness**: Consider adversarial training or certified defenses for high-risk deployments
2. **Fairness Gaps**: Post-processing calibration for [group] to reduce gap to <5%
3. **OOD Detection**: Add OOD detector module (e.g., Mahalanobis distance)

**Not Recommended**:
- Adversarial environments without mitigation
- Ultra-low latency requirements (<50ms)
```

#### 7.3 Quality Gate 2 Validation
Run evaluator agent:
```bash
npx claude-flow@alpha sparc run evaluator \
  "/validate-gate-2 --pipeline E --evaluation-report docs/holistic_evaluation_report.md"
```

**Gate 2 Requirements**:
- [ ] Holistic evaluation complete across 6+ dimensions
- [ ] Accuracy meets or exceeds baseline
- [ ] Fairness evaluation PASS (demographic parity <10%)
- [ ] Safety evaluation PASS (ethics-agent APPROVED)
- [ ] Efficiency acceptable (latency, memory, energy within targets)
- [ ] Interpretability analysis complete
- [ ] Trade-offs documented
- [ ] Deployment recommendations provided

#### 7.4 Gate 2 Decision
Based on evaluator's assessment:

**APPROVED**: All dimensions PASS, proceed to archival (Gate 3)
**CONDITIONAL**: Minor gaps with mitigation plan, proceed with restrictions
**REJECTED**: Critical issues (e.g., unmitigated safety risks), return to method development

#### 7.5 Memory Storage
```bash
npx claude-flow@alpha memory store \
  --key "sop/gate-2/holistic-evaluation" \
  --value "$(cat docs/holistic_evaluation_report.md)" \
  --metadata '{"status": "APPROVED", "dimensions": 6, "date": "2025-11-01"}'
```

**Deliverable**: Quality Gate 2 decision and holistic evaluation report

---

## Integration with Deep Research SOP

### Pipeline Integration
- **Pipeline E (Holistic Evaluation)**: This skill implements comprehensive model evaluation
- **Prerequisite**: Method development complete
- **Next Step**: Archival and reproducibility packaging (Gate 3) if APPROVED

### Quality Gates
- **Gate 2**: This skill is REQUIRED for Gate 2 validation
- **Gate 3**: Holistic evaluation report included in reproducibility package

### Agent Coordination
```
Flow: tester → ethics-agent → archivist → evaluator

Phase 1-5: tester performs accuracy, robustness, efficiency, interpretability evaluations
Phase 2, 6: ethics-agent conducts fairness and safety evaluations
Phase 7: evaluator synthesizes results and validates Gate 2
archivist: Stores holistic evaluation report in reproducibility package
```

### Memory Coordination
```bash
# Store evaluation results for future reference
npx claude-flow@alpha memory store \
  --key "sop/holistic-evaluation/results" \
  --value "$(cat experiments/results/holistic_evaluation/summary.json)"

# Retrieve baseline evaluation for comparison
npx claude-flow@alpha memory retrieve \
  --key "sop/baseline-replication/evaluation"
```

---

## Troubleshooting

### Issue: Fairness evaluation fails (demographic parity >10%)
**Symptoms**: Demographic parity difference exceeds 10% threshold
**Solutions**:
```bash
# Post-processing calibration
python scripts/fairness_calibration.py \
  --model experiments/results/best_checkpoint.pth \
  --method "equalized_odds_postprocessing" \
  --sensitive-attribute race \
  --output experiments/results/calibrated_model.pth

# Re-evaluate
python scripts/fairness_eval.py --model experiments/results/calibrated_model.pth
```

### Issue: Adversarial robustness extremely low (<30%)
**Symptoms**: Model highly vulnerable to adversarial attacks
**Solutions**:
1. Adversarial training (retrain with adversarial examples)
2. Certified defenses (randomized smoothing, interval bound propagation)
3. Ensemble methods (combine multiple models)
4. Input preprocessing (JPEG compression, bit depth reduction)

**Trade-off**: Adversarial training may reduce clean accuracy by 2-5%

### Issue: OOD detection poor (AUROC <0.70)
**Symptoms**: Model cannot distinguish in-distribution from out-of-distribution
**Solutions**:
```bash
# Add OOD detection module
python scripts/add_ood_detector.py \
  --model experiments/results/best_checkpoint.pth \
  --method "mahalanobis_distance" \
  --calibration-data val_dataset \
  --output experiments/results/model_with_ood.pth
```

### Issue: Gate 2 validation rejected due to ethics review
**Symptoms**: ethics-agent flags critical safety or fairness risks
**Solutions**:
```bash
# Review ethics assessment
npx claude-flow@alpha sparc run ethics-agent \
  "/assess-risks --component model --gate 2 --verbose"

# Address critical risks
# - Safety: Implement content filtering, adversarial prompt detection
# - Fairness: Bias mitigation techniques (reweighting, calibration)
# - Privacy: Differential privacy training, federated learning
```

---

## Related Skills and Commands

### Prerequisites
- `method-development` - Must complete before holistic evaluation

### Next Steps (after Gate 2 APPROVED)
- `reproducibility-audit` - Audit reproducibility before archival
- `deployment-readiness` - Prepare model for production deployment

### Related Commands
- `/validate-gate-2` - Gate 2 validation (evaluator agent)
- `/assess-risks` - Ethics and safety review (ethics-agent)
- `/safety-eval` - Safety-specific evaluation (ethics-agent)

### Parallel Skills
- Can run in parallel with literature synthesis (no dependencies)

---

## References

### Evaluation Frameworks
- Fairness Indicators (Google): https://github.com/tensorflow/fairness-indicators
- Foolbox (Adversarial Robustness): https://github.com/bethgelab/foolbox
- SHAP (Interpretability): https://github.com/slundberg/shap

### Academic Standards
- Gebru et al. (2021): Datasheets for Datasets
- Mitchell et al. (2019): Model Cards for Model Reporting
- Mehrabi et al. (2021): A Survey on Bias and Fairness in Machine Learning

### Fairness Metrics
- Demographic Parity: P(Y_hat=1 | A=a) equal across groups
- Equalized Odds: TPR and FPR equal across groups
- Calibration: P(Y=1 | Y_hat=p) = p across groups

### Robustness Standards
- RobustBench Leaderboard: https://robustbench.github.io/
- ImageNet-C (Corruption Robustness): Hendrycks & Dietterich (2019)

---

## Appendix

### Example Holistic Evaluation Summary

```
Model: Multi-Scale Attention ResNet-50
Date: 2025-11-01
Evaluator: tester + ethics-agent + evaluator

┌─────────────────────┬─────────────┬──────────┬──────────────────┐
│ Dimension           │ Metric      │ Value    │ Status           │
├─────────────────────┼─────────────┼──────────┼──────────────────┤
│ Accuracy            │ Test Acc    │ 87.5%    │ ✅ PASS (>85%)   │
│ Fairness            │ Dem. Parity │ 8.2%     │ ✅ PASS (<10%)   │
│ Robustness (Adv)    │ PGD ε=0.03  │ 45.8%    │ ⚠️  CONDITIONAL  │
│ Robustness (OOD)    │ AUROC       │ 0.82     │ ✅ PASS (>0.80)  │
│ Efficiency (Latency)│ Batch=32    │ 142.7ms  │ ✅ PASS (<200ms) │
│ Interpretability    │ SHAP        │ Complete │ ✅ PASS          │
│ Safety              │ Harmful Out │ 0.02%    │ ✅ PASS (<0.05%) │
└─────────────────────┴─────────────┴──────────┴──────────────────┘

Overall: CONDITIONAL APPROVAL
- Proceed to Gate 3 with adversarial robustness mitigation plan
- Deploy only in non-adversarial environments without mitigation
- Monitor fairness metrics in production
```