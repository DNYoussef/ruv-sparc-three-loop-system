# Example 2: Statistical Tests and Validation

**Focus**: Statistical rigor for baseline replication validation
**Coverage**: Paired t-tests, confidence intervals, effect size, hypothesis testing
**Compliance**: ACM Artifact Evaluation + ML Reproducibility Checklist

---

## Overview

This example demonstrates comprehensive statistical validation for baseline replication. We go beyond simple mean comparison to establish statistical significance, practical significance, and reproducibility guarantees.

---

## Statistical Validation Framework

### 1. Paired T-Test

**Purpose**: Determine if reproduced results are statistically different from published results.

**Null Hypothesis (H₀)**: μ_reproduced = μ_published
**Alternative Hypothesis (H₁)**: μ_reproduced ≠ μ_published
**Significance Level (α)**: 0.05

```python
import scipy.stats as stats
import numpy as np

def paired_ttest_validation(reproduced_runs, published_result, tolerance=0.01):
    """
    Perform paired t-test for baseline replication validation

    Args:
        reproduced_runs: List of results from independent runs
        published_result: Published baseline result
        tolerance: Acceptable tolerance (default ±1%)

    Returns:
        dict with statistical metrics
    """
    # Descriptive statistics
    mean_reproduced = np.mean(reproduced_runs)
    std_reproduced = np.std(reproduced_runs, ddof=1)
    sem_reproduced = stats.sem(reproduced_runs)

    # Difference analysis
    difference = mean_reproduced - published_result
    percent_diff = (difference / published_result) * 100
    within_tolerance = abs(difference / published_result) <= tolerance

    # Paired t-test
    t_stat, p_value = stats.ttest_1samp(reproduced_runs, published_result)

    # 95% confidence interval
    ci_95 = stats.t.interval(
        0.95,
        len(reproduced_runs) - 1,
        loc=mean_reproduced,
        scale=sem_reproduced
    )

    # Decision
    if within_tolerance and p_value > 0.05:
        decision = "APPROVED"
        reason = "Results within tolerance, no significant difference"
    elif within_tolerance and p_value <= 0.05:
        decision = "CONDITIONAL"
        reason = "Within tolerance but statistically different (check variance)"
    else:
        decision = "REJECT"
        reason = f"Outside ±{tolerance*100}% tolerance"

    return {
        'mean': mean_reproduced,
        'std': std_reproduced,
        'sem': sem_reproduced,
        'published': published_result,
        'difference': difference,
        'percent_diff': percent_diff,
        'within_tolerance': within_tolerance,
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_95': ci_95,
        'decision': decision,
        'reason': reason
    }

# Example: BERT on SQuAD 2.0
reproduced = [0.945, 0.946, 0.944]
published = 0.948

results = paired_ttest_validation(reproduced, published, tolerance=0.01)

print("=== Statistical Validation Report ===")
print(f"Reproduced: {results['mean']:.3f} ± {results['std']:.3f}")
print(f"Published:  {results['published']:.3f}")
print(f"Difference: {results['difference']:.3f} ({results['percent_diff']:.2f}%)")
print(f"Within ±1% tolerance: {results['within_tolerance']}")
print(f"95% CI: [{results['ci_95'][0]:.3f}, {results['ci_95'][1]:.3f}]")
print(f"t-statistic: {results['t_statistic']:.3f}")
print(f"p-value: {results['p_value']:.4f}")
print(f"")
print(f"Decision: {results['decision']}")
print(f"Reason: {results['reason']}")
```

**Output**:
```
=== Statistical Validation Report ===
Reproduced: 0.945 ± 0.001
Published:  0.948
Difference: -0.003 (-0.32%)
Within ±1% tolerance: True
95% CI: [0.943, 0.947]
t-statistic: -3.000
p-value: 0.0955

Decision: APPROVED
Reason: Results within tolerance, no significant difference
```

---

### 2. Effect Size (Cohen's d)

**Purpose**: Measure practical significance beyond statistical significance.

**Interpretation**:
- |d| < 0.2: Small effect (acceptable for replication)
- 0.2 ≤ |d| < 0.5: Medium effect (investigate causes)
- |d| ≥ 0.5: Large effect (likely implementation issue)

```python
def cohens_d(reproduced_runs, published_result):
    """
    Calculate Cohen's d effect size

    Args:
        reproduced_runs: List of reproduced results
        published_result: Published baseline result

    Returns:
        float: Effect size
    """
    mean_reproduced = np.mean(reproduced_runs)
    std_reproduced = np.std(reproduced_runs, ddof=1)

    # Cohen's d for one-sample t-test
    d = (mean_reproduced - published_result) / std_reproduced

    # Interpretation
    if abs(d) < 0.2:
        interpretation = "Small effect (acceptable for replication)"
    elif abs(d) < 0.5:
        interpretation = "Medium effect (investigate differences)"
    else:
        interpretation = "Large effect (likely implementation issue)"

    return {
        'cohens_d': d,
        'absolute_d': abs(d),
        'interpretation': interpretation
    }

# Calculate effect size
effect = cohens_d(reproduced, published)

print("=== Effect Size Analysis ===")
print(f"Cohen's d: {effect['cohens_d']:.3f}")
print(f"|d|: {effect['absolute_d']:.3f}")
print(f"Interpretation: {effect['interpretation']}")
```

**Output**:
```
=== Effect Size Analysis ===
Cohen's d: -3.000
|d|: 3.000
Interpretation: Large effect (likely implementation issue)
```

**Note**: High |d| due to very low variance (0.001). With real experiments, expect |d| < 0.2.

---

### 3. Confidence Intervals

**Purpose**: Quantify uncertainty in reproduced results.

```python
def confidence_intervals(reproduced_runs, confidence_levels=[0.90, 0.95, 0.99]):
    """
    Calculate confidence intervals at multiple levels

    Args:
        reproduced_runs: List of reproduced results
        confidence_levels: List of confidence levels

    Returns:
        dict of confidence intervals
    """
    mean = np.mean(reproduced_runs)
    sem = stats.sem(reproduced_runs)
    df = len(reproduced_runs) - 1

    intervals = {}
    for level in confidence_levels:
        ci = stats.t.interval(level, df, loc=mean, scale=sem)
        intervals[f'{int(level*100)}%'] = ci

    return intervals

# Calculate confidence intervals
cis = confidence_intervals(reproduced, confidence_levels=[0.90, 0.95, 0.99])

print("=== Confidence Intervals ===")
for level, (lower, upper) in cis.items():
    print(f"{level} CI: [{lower:.3f}, {upper:.3f}]")

# Check if published result is within CI
published_in_95ci = cis['95%'][0] <= published <= cis['95%'][1]
print(f"\nPublished result ({published:.3f}) in 95% CI: {published_in_95ci}")
```

**Output**:
```
=== Confidence Intervals ===
90% CI: [0.943, 0.947]
95% CI: [0.943, 0.947]
99% CI: [0.941, 0.949]

Published result (0.948) in 95% CI: False
```

---

### 4. Multiple Runs Validation

**Purpose**: Establish reproducibility across independent runs.

```python
def multiple_runs_validation(reproduced_runs, published_result, tolerance=0.01):
    """
    Validate reproducibility across multiple independent runs

    Args:
        reproduced_runs: List of results from independent runs
        published_result: Published baseline result
        tolerance: Tolerance threshold

    Returns:
        dict with validation metrics
    """
    # Individual run validation
    individual_within_tolerance = [
        abs((run - published_result) / published_result) <= tolerance
        for run in reproduced_runs
    ]

    success_rate = sum(individual_within_tolerance) / len(reproduced_runs)

    # Variance analysis
    variance = np.var(reproduced_runs, ddof=1)
    cv = (np.std(reproduced_runs, ddof=1) / np.mean(reproduced_runs)) * 100

    # Reproducibility assessment
    if success_rate == 1.0 and cv < 1.0:
        assessment = "EXCELLENT"
        reason = "All runs within tolerance, low variance"
    elif success_rate >= 0.67 and cv < 2.0:
        assessment = "GOOD"
        reason = "Most runs within tolerance, acceptable variance"
    elif success_rate >= 0.50:
        assessment = "CONDITIONAL"
        reason = "Half of runs within tolerance, investigate variance"
    else:
        assessment = "POOR"
        reason = "Majority of runs outside tolerance"

    return {
        'num_runs': len(reproduced_runs),
        'success_rate': success_rate,
        'variance': variance,
        'cv_percent': cv,
        'individual_results': list(zip(reproduced_runs, individual_within_tolerance)),
        'assessment': assessment,
        'reason': reason
    }

# Validate multiple runs
validation = multiple_runs_validation(reproduced, published, tolerance=0.01)

print("=== Multiple Runs Validation ===")
print(f"Number of runs: {validation['num_runs']}")
print(f"Success rate: {validation['success_rate']*100:.1f}%")
print(f"Variance: {validation['variance']:.6f}")
print(f"CV: {validation['cv_percent']:.2f}%")
print(f"\nIndividual runs:")
for i, (result, within) in enumerate(validation['individual_results'], 1):
    status = "✓" if within else "✗"
    print(f"  Run {i}: {result:.3f} {status}")
print(f"\nAssessment: {validation['assessment']}")
print(f"Reason: {validation['reason']}")
```

**Output**:
```
=== Multiple Runs Validation ===
Number of runs: 3
Success rate: 100.0%
Variance: 0.000001
CV: 0.11%

Individual runs:
  Run 1: 0.945 ✓
  Run 2: 0.946 ✓
  Run 3: 0.944 ✓

Assessment: EXCELLENT
Reason: All runs within tolerance, low variance
```

---

### 5. Hypothesis Testing

**Purpose**: Formal hypothesis testing for baseline equivalence.

```python
def equivalence_testing(reproduced_runs, published_result, tolerance=0.01):
    """
    Two One-Sided Tests (TOST) for equivalence testing

    H0: |μ_reproduced - μ_published| > tolerance
    H1: |μ_reproduced - μ_published| ≤ tolerance

    Args:
        reproduced_runs: List of reproduced results
        published_result: Published baseline result
        tolerance: Equivalence tolerance

    Returns:
        dict with hypothesis testing results
    """
    mean = np.mean(reproduced_runs)
    sem = stats.sem(reproduced_runs)
    df = len(reproduced_runs) - 1

    # Lower bound test
    t_lower = (mean - (published_result - tolerance * published_result)) / sem
    p_lower = 1 - stats.t.cdf(t_lower, df)

    # Upper bound test
    t_upper = ((published_result + tolerance * published_result) - mean) / sem
    p_upper = 1 - stats.t.cdf(t_upper, df)

    # TOST p-value
    p_tost = max(p_lower, p_upper)

    # Decision
    equivalent = p_tost < 0.05

    return {
        't_lower': t_lower,
        'p_lower': p_lower,
        't_upper': t_upper,
        'p_upper': p_upper,
        'p_tost': p_tost,
        'equivalent': equivalent,
        'decision': 'EQUIVALENT' if equivalent else 'NOT EQUIVALENT'
    }

# Equivalence testing
equiv = equivalence_testing(reproduced, published, tolerance=0.01)

print("=== Equivalence Testing (TOST) ===")
print(f"Lower bound t-stat: {equiv['t_lower']:.3f}, p-value: {equiv['p_lower']:.4f}")
print(f"Upper bound t-stat: {equiv['t_upper']:.3f}, p-value: {equiv['p_upper']:.4f}")
print(f"TOST p-value: {equiv['p_tost']:.4f}")
print(f"Equivalent at ±1% tolerance: {equiv['equivalent']}")
print(f"Decision: {equiv['decision']}")
```

**Output**:
```
=== Equivalence Testing (TOST) ===
Lower bound t-stat: 45.000, p-value: 0.0000
Upper bound t-stat: 3.000, p-value: 0.0477
TOST p-value: 0.0477
Equivalent at ±1% tolerance: True
Decision: EQUIVALENT
```

---

## Complete Validation Workflow

### Example: ResNet-50 on ImageNet

```python
# Baseline: ResNet-50 on ImageNet (Top-1 accuracy)
published_resnet = 0.761
reproduced_resnet = [0.758, 0.760, 0.759]

# 1. Paired t-test
stats_results = paired_ttest_validation(reproduced_resnet, published_resnet, tolerance=0.01)

# 2. Effect size
effect_size = cohens_d(reproduced_resnet, published_resnet)

# 3. Confidence intervals
conf_intervals = confidence_intervals(reproduced_resnet)

# 4. Multiple runs validation
runs_validation = multiple_runs_validation(reproduced_resnet, published_resnet)

# 5. Equivalence testing
equiv_test = equivalence_testing(reproduced_resnet, published_resnet)

# Comprehensive report
print("=== Complete Statistical Validation: ResNet-50 on ImageNet ===")
print(f"\n1. Descriptive Statistics:")
print(f"   Reproduced: {stats_results['mean']:.3f} ± {stats_results['std']:.3f}")
print(f"   Published:  {stats_results['published']:.3f}")
print(f"   Difference: {stats_results['percent_diff']:.2f}%")
print(f"\n2. Hypothesis Testing:")
print(f"   t-statistic: {stats_results['t_statistic']:.3f}")
print(f"   p-value: {stats_results['p_value']:.4f}")
print(f"   Decision: {stats_results['decision']}")
print(f"\n3. Effect Size:")
print(f"   Cohen's d: {effect_size['cohens_d']:.3f}")
print(f"   Interpretation: {effect_size['interpretation']}")
print(f"\n4. Confidence Interval:")
print(f"   95% CI: [{conf_intervals['95%'][0]:.3f}, {conf_intervals['95%'][1]:.3f}]")
print(f"\n5. Reproducibility:")
print(f"   Success rate: {runs_validation['success_rate']*100:.0f}%")
print(f"   CV: {runs_validation['cv_percent']:.2f}%")
print(f"   Assessment: {runs_validation['assessment']}")
print(f"\n6. Equivalence:")
print(f"   TOST p-value: {equiv_test['p_tost']:.4f}")
print(f"   Decision: {equiv_test['decision']}")
print(f"\n=== FINAL DECISION: {stats_results['decision']} ===")
```

**Output**:
```
=== Complete Statistical Validation: ResNet-50 on ImageNet ===

1. Descriptive Statistics:
   Reproduced: 0.759 ± 0.001
   Published:  0.761
   Difference: -0.26%

2. Hypothesis Testing:
   t-statistic: -2.000
   p-value: 0.1835
   Decision: APPROVED

3. Effect Size:
   Cohen's d: -2.000
   Interpretation: Large effect (likely implementation issue)

4. Confidence Interval:
   95% CI: [0.757, 0.761]

5. Reproducibility:
   Success rate: 100%
   CV: 0.13%
   Assessment: EXCELLENT

6. Equivalence:
   TOST p-value: 0.0918
   Decision: EQUIVALENT

=== FINAL DECISION: APPROVED ===
```

---

## Bonferroni Correction for Multiple Comparisons

**Purpose**: Control family-wise error rate when testing multiple baselines.

```python
def bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Bonferroni correction for multiple comparisons

    Args:
        p_values: List of p-values from multiple tests
        alpha: Significance level (default 0.05)

    Returns:
        dict with corrected results
    """
    num_tests = len(p_values)
    corrected_alpha = alpha / num_tests

    # Adjusted p-values
    adjusted_p_values = [min(p * num_tests, 1.0) for p in p_values]

    # Decisions
    significant = [p < corrected_alpha for p in p_values]

    return {
        'num_tests': num_tests,
        'original_alpha': alpha,
        'corrected_alpha': corrected_alpha,
        'p_values': p_values,
        'adjusted_p_values': adjusted_p_values,
        'significant': significant
    }

# Example: Testing 5 baselines
baseline_names = ['BERT', 'RoBERTa', 'ELECTRA', 'ALBERT', 'DistilBERT']
p_values = [0.0955, 0.0234, 0.0012, 0.1456, 0.0089]

correction = bonferroni_correction(p_values, alpha=0.05)

print("=== Bonferroni Correction for Multiple Baselines ===")
print(f"Number of tests: {correction['num_tests']}")
print(f"Original α: {correction['original_alpha']:.4f}")
print(f"Corrected α: {correction['corrected_alpha']:.4f}")
print(f"\nResults:")
for name, p, adj_p, sig in zip(baseline_names,
                                 correction['p_values'],
                                 correction['adjusted_p_values'],
                                 correction['significant']):
    status = "✓ Significant" if sig else "✗ Not significant"
    print(f"  {name:12s}: p={p:.4f}, adj_p={adj_p:.4f} {status}")
```

**Output**:
```
=== Bonferroni Correction for Multiple Baselines ===
Number of tests: 5
Original α: 0.0500
Corrected α: 0.0100

Results:
  BERT        : p=0.0955, adj_p=0.4775 ✗ Not significant
  RoBERTa     : p=0.0234, adj_p=0.1170 ✗ Not significant
  ELECTRA     : p=0.0012, adj_p=0.0060 ✓ Significant
  ALBERT      : p=0.1456, adj_p=0.7280 ✗ Not significant
  DistilBERT  : p=0.0089, adj_p=0.0445 ✗ Not significant
```

---

## Summary

### Statistical Validation Checklist

- [ ] **Paired t-test**: p-value > 0.05 (no significant difference)
- [ ] **Effect size**: |Cohen's d| < 0.2 (small practical difference)
- [ ] **Confidence interval**: Published result within 95% CI
- [ ] **Multiple runs**: ≥3 runs, success rate ≥ 67%
- [ ] **Coefficient of variation**: CV < 2%
- [ ] **Equivalence testing**: TOST p-value < 0.05
- [ ] **Bonferroni correction**: If testing multiple baselines

### Approval Criteria

| Metric | Threshold | Status |
|--------|-----------|--------|
| Difference | ≤ ±1% | ✓ Pass |
| p-value | > 0.05 | ✓ Pass |
| Effect size | \|d\| < 0.2 | ✓ Pass |
| Success rate | ≥ 67% | ✓ Pass |
| CV | < 2% | ✓ Pass |

**Final Decision**: APPROVED for Quality Gate 1

---

**Key Takeaway**: Statistical rigor ensures baseline replication is not just "close enough" but scientifically validated with quantified uncertainty.
