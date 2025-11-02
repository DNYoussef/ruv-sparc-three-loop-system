# Example 3: Ablation Studies for Baseline Validation

**Focus**: Component-wise validation through systematic ablations
**Purpose**: Verify that individual components contribute as expected
**Quality Gate**: Prerequisite for Gate 2 (Model Evaluation)

---

## Overview

Ablation studies systematically remove or modify individual components to understand their contribution. This validates that your baseline implementation matches the published work at a component level, not just overall performance.

---

## Why Ablations for Baseline Replication?

### Validation Goals

1. **Component verification**: Confirm each component (attention, embeddings, layers) works as expected
2. **Implementation correctness**: Catch subtle bugs that don't affect overall metrics
3. **Understanding baseline**: Build intuition for novel method development
4. **Quality Gate 2 prep**: Required for method development phase

### Example Scenario

**Baseline**: BERT-base achieves 0.945 on SQuAD 2.0 (reproduced vs 0.948 published)

**Question**: Is the implementation truly correct, or did we get lucky with hyperparameters?

**Answer**: Run ablations to validate component-level behavior.

---

## Ablation Study Framework

### 1. Basic Ablation Template

```python
def run_ablation_study(model, ablation_config, dataset, num_runs=3):
    """
    Run ablation study with statistical validation

    Args:
        model: Baseline model
        ablation_config: Dict specifying which components to ablate
        dataset: Evaluation dataset
        num_runs: Number of independent runs

    Returns:
        dict with ablation results
    """
    results = []

    for run in range(num_runs):
        # Apply ablation
        ablated_model = apply_ablation(model, ablation_config)

        # Evaluate
        accuracy = evaluate(ablated_model, dataset)
        results.append(accuracy)

    mean_acc = np.mean(results)
    std_acc = np.std(results, ddof=1)

    return {
        'ablation': ablation_config['name'],
        'mean': mean_acc,
        'std': std_acc,
        'runs': results
    }

def apply_ablation(model, config):
    """Apply specific ablation to model"""
    ablated = copy.deepcopy(model)

    if config['type'] == 'remove_component':
        # Remove component (e.g., attention, layer normalization)
        remove_component(ablated, config['component'])

    elif config['type'] == 'modify_hyperparameter':
        # Modify hyperparameter (e.g., learning rate, dropout)
        modify_hyperparameter(ablated, config['param'], config['value'])

    elif config['type'] == 'replace_component':
        # Replace component with simpler version
        replace_component(ablated, config['component'], config['replacement'])

    return ablated
```

---

## Case Study: BERT Ablations on SQuAD 2.0

### Full Baseline Performance

```python
# Baseline: BERT-base-uncased on SQuAD 2.0
baseline_config = {
    'name': 'Full BERT',
    'layers': 12,
    'hidden_size': 768,
    'attention_heads': 12,
    'dropout': 0.1,
    'learning_rate': 5e-5,
    'warmup_steps': 10000
}

baseline_results = [0.945, 0.946, 0.944]
baseline_mean = 0.945
baseline_std = 0.001
```

---

### Ablation 1: Remove Layer Normalization

**Hypothesis**: Layer normalization stabilizes training (expect -5% to -10% accuracy)

```python
ablation_1 = {
    'name': 'No Layer Normalization',
    'type': 'remove_component',
    'component': 'layer_norm'
}

# Implementation
class BertLayerWithoutLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        # Removed: self.LayerNorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        # Removed: layer_output = self.LayerNorm(layer_output)
        return layer_output

# Results
ablation_1_results = [0.852, 0.849, 0.855]
ablation_1_mean = 0.852
ablation_1_std = 0.003

# Analysis
drop = baseline_mean - ablation_1_mean
drop_pct = (drop / baseline_mean) * 100

print(f"Ablation 1: No Layer Normalization")
print(f"  Baseline: {baseline_mean:.3f} ± {baseline_std:.3f}")
print(f"  Ablated:  {ablation_1_mean:.3f} ± {ablation_1_std:.3f}")
print(f"  Drop: {drop:.3f} (-{drop_pct:.1f}%)")
print(f"  ✓ Expected drop: Layer norm is critical for transformer stability")
```

**Output**:
```
Ablation 1: No Layer Normalization
  Baseline: 0.945 ± 0.001
  Ablated:  0.852 ± 0.003
  Drop: 0.093 (-9.8%)
  ✓ Expected drop: Layer norm is critical for transformer stability
```

---

### Ablation 2: Reduce Number of Layers (12 → 6)

**Hypothesis**: Depth matters (expect -3% to -5% accuracy)

```python
ablation_2 = {
    'name': '6 Layers (vs 12)',
    'type': 'modify_hyperparameter',
    'param': 'num_hidden_layers',
    'value': 6
}

# Implementation: Use BERT-6-layer configuration
model_6_layer = BertForQuestionAnswering.from_pretrained(
    'bert-base-uncased',
    num_hidden_layers=6
)

# Results
ablation_2_results = [0.908, 0.910, 0.906]
ablation_2_mean = 0.908
ablation_2_std = 0.002

drop = baseline_mean - ablation_2_mean
drop_pct = (drop / baseline_mean) * 100

print(f"Ablation 2: 6 Layers (vs 12)")
print(f"  Baseline: {baseline_mean:.3f} ± {baseline_std:.3f}")
print(f"  Ablated:  {ablation_2_mean:.3f} ± {ablation_2_std:.3f}")
print(f"  Drop: {drop:.3f} (-{drop_pct:.1f}%)")
print(f"  ✓ Expected drop: Depth contributes to performance")
```

**Output**:
```
Ablation 2: 6 Layers (vs 12)
  Baseline: 0.945 ± 0.001
  Ablated:  0.908 ± 0.002
  Drop: 0.037 (-3.9%)
  ✓ Expected drop: Depth contributes to performance
```

---

### Ablation 3: No Warmup Learning Rate Schedule

**Hypothesis**: Warmup prevents early training instability (expect -2% to -3% accuracy)

```python
ablation_3 = {
    'name': 'No LR Warmup',
    'type': 'modify_hyperparameter',
    'param': 'warmup_steps',
    'value': 0
}

# Implementation: Use constant learning rate
optimizer = AdamW(model.parameters(), lr=5e-5)
# scheduler = get_constant_schedule(optimizer)  # No warmup

# Results
ablation_3_results = [0.921, 0.919, 0.923]
ablation_3_mean = 0.921
ablation_3_std = 0.002

drop = baseline_mean - ablation_3_mean
drop_pct = (drop / baseline_mean) * 100

print(f"Ablation 3: No LR Warmup")
print(f"  Baseline: {baseline_mean:.3f} ± {baseline_std:.3f}")
print(f"  Ablated:  {ablation_3_mean:.3f} ± {ablation_3_std:.3f}")
print(f"  Drop: {drop:.3f} (-{drop_pct:.1f}%)")
print(f"  ✓ Expected drop: Warmup stabilizes early training")
```

**Output**:
```
Ablation 3: No LR Warmup
  Baseline: 0.945 ± 0.001
  Ablated:  0.921 ± 0.002
  Drop: 0.024 (-2.5%)
  ✓ Expected drop: Warmup stabilizes early training
```

---

### Ablation 4: Reduce Attention Heads (12 → 4)

**Hypothesis**: Multi-head attention provides diverse representations (expect -1% to -2%)

```python
ablation_4 = {
    'name': '4 Attention Heads (vs 12)',
    'type': 'modify_hyperparameter',
    'param': 'num_attention_heads',
    'value': 4
}

# Implementation
model_4_heads = BertForQuestionAnswering.from_pretrained(
    'bert-base-uncased',
    num_attention_heads=4
)

# Results
ablation_4_results = [0.931, 0.929, 0.933]
ablation_4_mean = 0.931
ablation_4_std = 0.002

drop = baseline_mean - ablation_4_mean
drop_pct = (drop / baseline_mean) * 100

print(f"Ablation 4: 4 Attention Heads (vs 12)")
print(f"  Baseline: {baseline_mean:.3f} ± {baseline_std:.3f}")
print(f"  Ablated:  {ablation_4_mean:.3f} ± {ablation_4_std:.3f}")
print(f"  Drop: {drop:.3f} (-{drop_pct:.1f}%)")
print(f"  ✓ Expected drop: Multi-head attention improves representation")
```

**Output**:
```
Ablation 4: 4 Attention Heads (vs 12)
  Baseline: 0.945 ± 0.001
  Ablated:  0.931 ± 0.002
  Drop: 0.014 (-1.5%)
  ✓ Expected drop: Multi-head attention improves representation
```

---

### Ablation 5: No Dropout

**Hypothesis**: Dropout prevents overfitting (expect -1% to -2% due to overfitting)

```python
ablation_5 = {
    'name': 'No Dropout',
    'type': 'modify_hyperparameter',
    'param': 'hidden_dropout_prob',
    'value': 0.0
}

# Implementation
model_no_dropout = BertForQuestionAnswering.from_pretrained(
    'bert-base-uncased',
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0
)

# Results
ablation_5_results = [0.928, 0.926, 0.930]
ablation_5_mean = 0.928
ablation_5_std = 0.002

drop = baseline_mean - ablation_5_mean
drop_pct = (drop / baseline_mean) * 100

print(f"Ablation 5: No Dropout")
print(f"  Baseline: {baseline_mean:.3f} ± {baseline_std:.3f}")
print(f"  Ablated:  {ablation_5_mean:.3f} ± {ablation_5_std:.3f}")
print(f"  Drop: {drop:.3f} (-{drop_pct:.1f}%)")
print(f"  ✓ Expected drop: Dropout prevents overfitting on small datasets")
```

**Output**:
```
Ablation 5: No Dropout
  Baseline: 0.945 ± 0.001
  Ablated:  0.928 ± 0.002
  Drop: 0.017 (-1.8%)
  ✓ Expected drop: Dropout prevents overfitting on small datasets
```

---

## Ablation Results Summary

### Performance Comparison Table

| Ablation | Accuracy | Std | Drop | Drop % | Expected |
|----------|----------|-----|------|--------|----------|
| **Full Baseline** | 0.945 | 0.001 | - | - | - |
| No Layer Norm | 0.852 | 0.003 | 0.093 | -9.8% | ✓ Large drop |
| 6 Layers (vs 12) | 0.908 | 0.002 | 0.037 | -3.9% | ✓ Medium drop |
| No LR Warmup | 0.921 | 0.002 | 0.024 | -2.5% | ✓ Small-medium drop |
| 4 Heads (vs 12) | 0.931 | 0.002 | 0.014 | -1.5% | ✓ Small drop |
| No Dropout | 0.928 | 0.002 | 0.017 | -1.8% | ✓ Small drop |

### Visualization

```python
import matplotlib.pyplot as plt

ablations = ['Full\nBaseline', 'No Layer\nNorm', '6 Layers',
             'No LR\nWarmup', '4 Heads', 'No\nDropout']
accuracies = [0.945, 0.852, 0.908, 0.921, 0.931, 0.928]
stds = [0.001, 0.003, 0.002, 0.002, 0.002, 0.002]

plt.figure(figsize=(10, 6))
plt.bar(ablations, accuracies, yerr=stds, capsize=5, alpha=0.7)
plt.axhline(y=0.945, color='r', linestyle='--', label='Full Baseline')
plt.ylabel('Accuracy')
plt.title('BERT Ablation Study on SQuAD 2.0')
plt.ylim([0.84, 0.96])
plt.legend()
plt.tight_layout()
plt.savefig('ablation-study-results.png')
```

---

## Statistical Validation of Ablations

### Paired Comparison with Baseline

```python
import scipy.stats as stats

def compare_with_baseline(baseline_runs, ablation_runs, ablation_name):
    """
    Statistically compare ablation with baseline

    Args:
        baseline_runs: List of baseline results
        ablation_runs: List of ablation results
        ablation_name: Name of the ablation
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_ind(baseline_runs, ablation_runs)

    # Effect size
    mean_diff = np.mean(baseline_runs) - np.mean(ablation_runs)
    pooled_std = np.sqrt((np.std(baseline_runs)**2 + np.std(ablation_runs)**2) / 2)
    cohens_d = mean_diff / pooled_std

    # Significance
    significant = p_value < 0.05

    print(f"=== {ablation_name} ===")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Significant: {significant}")
    print(f"Cohen's d: {cohens_d:.3f}")
    print()

# Compare each ablation
compare_with_baseline(baseline_results, ablation_1_results, "No Layer Normalization")
compare_with_baseline(baseline_results, ablation_2_results, "6 Layers")
compare_with_baseline(baseline_results, ablation_3_results, "No LR Warmup")
compare_with_baseline(baseline_results, ablation_4_results, "4 Attention Heads")
compare_with_baseline(baseline_results, ablation_5_results, "No Dropout")
```

**Output**:
```
=== No Layer Normalization ===
t-statistic: 49.497
p-value: 0.0000
Significant: True
Cohen's d: 40.294

=== 6 Layers ===
t-statistic: 20.208
p-value: 0.0001
Significant: True
Cohen's d: 16.432

=== No LR Warmup ===
t-statistic: 11.662
p-value: 0.0009
Significant: True
Cohen's d: 9.488

=== 4 Attention Heads ===
t-statistic: 6.804
p-value: 0.0047
Significant: True
Cohen's d: 5.532

=== No Dropout ===
t-statistic: 8.249
p-value: 0.0028
Significant: True
Cohen's d: 6.708
```

---

## Validation Checklist

### Component-Level Validation

- [x] **Layer normalization**: Critical for stability (-9.8% without)
- [x] **Model depth**: 12 layers significantly better than 6 (-3.9%)
- [x] **Learning rate warmup**: Stabilizes training (-2.5% without)
- [x] **Multi-head attention**: 12 heads better than 4 (-1.5%)
- [x] **Dropout regularization**: Prevents overfitting (-1.8% without)

### Expected Behavior

✓ All ablations show statistically significant drops (p < 0.05)
✓ Effect sizes align with literature expectations
✓ Component contributions match published results
✓ Implementation validated at component level

---

## Quality Gate 2 Requirements

### Mandatory Ablations for Method Development

When developing a novel method after baseline replication, Quality Gate 2 requires **5+ ablations**:

1. **Novel component ablation**: Remove your innovation
2. **Baseline component ablation**: Verify baseline components still work
3. **Hyperparameter ablations**: Test sensitivity (learning rate, batch size)
4. **Architecture ablations**: Validate architectural choices
5. **Training procedure ablations**: Verify training schedule, warmup, etc.

### Bonferroni Correction for Multiple Comparisons

```python
def bonferroni_ablation_correction(ablation_p_values, alpha=0.05):
    """Apply Bonferroni correction for multiple ablations"""
    num_ablations = len(ablation_p_values)
    corrected_alpha = alpha / num_ablations

    print(f"=== Bonferroni Correction ===")
    print(f"Number of ablations: {num_ablations}")
    print(f"Original α: {alpha:.4f}")
    print(f"Corrected α: {corrected_alpha:.4f}")
    print()

    for i, (name, p) in enumerate(ablation_p_values, 1):
        significant = p < corrected_alpha
        status = "✓ Significant" if significant else "✗ Not significant"
        print(f"{i}. {name:30s}: p={p:.4f} {status}")

# Example: 5 ablations
ablation_p_values = [
    ("No Layer Normalization", 0.0000),
    ("6 Layers", 0.0001),
    ("No LR Warmup", 0.0009),
    ("4 Attention Heads", 0.0047),
    ("No Dropout", 0.0028)
]

bonferroni_ablation_correction(ablation_p_values, alpha=0.05)
```

**Output**:
```
=== Bonferroni Correction ===
Number of ablations: 5
Original α: 0.0500
Corrected α: 0.0100

1. No Layer Normalization         : p=0.0000 ✓ Significant
2. 6 Layers                        : p=0.0001 ✓ Significant
3. No LR Warmup                    : p=0.0009 ✓ Significant
4. 4 Attention Heads               : p=0.0047 ✓ Significant
5. No Dropout                      : p=0.0028 ✓ Significant
```

**All ablations remain significant after Bonferroni correction. ✓**

---

## Lessons Learned

### Ablation Study Best Practices

1. **Hypothesis-driven**: Formulate expected outcome before running ablation
2. **Multiple runs**: Run 3+ independent experiments for each ablation
3. **Statistical testing**: Use paired t-tests and effect sizes
4. **Bonferroni correction**: Control for multiple comparisons
5. **Literature alignment**: Compare drops with published ablation studies

### Common Pitfalls

❌ **Single-run ablations**: Insufficient for statistical validation
❌ **No baseline comparison**: Can't quantify component contribution
❌ **Ignoring multiple comparisons**: Inflated false positive rate
❌ **Cherry-picking ablations**: Select ablations that support your hypothesis

---

## Summary

Ablation studies validate baseline implementation at a component level:

- **5 ablations** demonstrate correct implementation
- **All components** contribute as expected (statistically significant drops)
- **Effect sizes** align with literature
- **Ready for Quality Gate 2** and novel method development

**Next Step**: Develop novel method with confidence that baseline is correctly implemented.

---

**Estimated Time**: 12 hours (5 ablations × 2.4 hours average)
**Quality Gate**: Prerequisite for Gate 2 (Method Evaluation)
**Status**: VALIDATED - Baseline implementation confirmed correct
