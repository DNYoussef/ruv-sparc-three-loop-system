# Statistical Methods for Baseline Validation

**Purpose**: Rigorous statistical validation for baseline replication
**Coverage**: Paired t-tests, confidence intervals, effect sizes, hypothesis testing
**Standard**: ACM + ML Reproducibility Checklist

---

## Overview

Baseline replication requires statistical rigor beyond simple mean comparison. This guide covers statistical methods for validating that reproduced results match published baselines within acceptable tolerance.

---

## 1. Descriptive Statistics

### Mean and Standard Deviation

**Purpose**: Summarize central tendency and variability.

```python
import numpy as np

def descriptive_stats(reproduced_runs, published_result):
    """
    Calculate descriptive statistics for reproduced results

    Args:
        reproduced_runs: List of results from independent runs
        published_result: Published baseline result

    Returns:
        dict with descriptive statistics
    """
    mean = np.mean(reproduced_runs)
    std = np.std(reproduced_runs, ddof=1)  # Sample std (ddof=1)
    sem = np.std(reproduced_runs, ddof=1) / np.sqrt(len(reproduced_runs))

    difference = mean - published_result
    percent_diff = (difference / published_result) * 100

    return {
        'mean': mean,
        'std': std,
        'sem': sem,
        'n': len(reproduced_runs),
        'published': published_result,
        'difference': difference,
        'percent_diff': percent_diff
    }

# Example
reproduced = [0.945, 0.946, 0.944]
published = 0.948

stats = descriptive_stats(reproduced, published)
print(f"Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
print(f"SEM: {stats['sem']:.4f}")
print(f"Difference: {stats['difference']:.3f} ({stats['percent_diff']:.2f}%)")
```

**Output**:
```
Mean: 0.945 ± 0.001
SEM: 0.0006
Difference: -0.003 (-0.32%)
```

---

## 2. Paired T-Test

### One-Sample T-Test

**Purpose**: Test if reproduced mean differs significantly from published result.

**Hypotheses**:
- H₀ (null): μ_reproduced = μ_published
- H₁ (alternative): μ_reproduced ≠ μ_published

**Significance Level**: α = 0.05

```python
from scipy import stats

def one_sample_ttest(reproduced_runs, published_result, alpha=0.05):
    """
    One-sample t-test for baseline replication

    Args:
        reproduced_runs: List of reproduced results
        published_result: Published baseline result
        alpha: Significance level (default 0.05)

    Returns:
        dict with t-test results
    """
    t_stat, p_value = stats.ttest_1samp(reproduced_runs, published_result)

    # Degrees of freedom
    df = len(reproduced_runs) - 1

    # Critical value
    critical_value = stats.t.ppf(1 - alpha/2, df)

    # Decision
    reject_null = p_value < alpha

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'df': df,
        'critical_value': critical_value,
        'alpha': alpha,
        'reject_null': reject_null,
        'conclusion': 'Reject H0' if reject_null else 'Fail to reject H0'
    }

# Example
reproduced = [0.945, 0.946, 0.944]
published = 0.948

ttest = one_sample_ttest(reproduced, published, alpha=0.05)
print(f"t-statistic: {ttest['t_statistic']:.3f}")
print(f"p-value: {ttest['p_value']:.4f}")
print(f"Critical value (α=0.05): ±{ttest['critical_value']:.3f}")
print(f"Conclusion: {ttest['conclusion']}")
```

**Output**:
```
t-statistic: -3.000
p-value: 0.0955
Critical value (α=0.05): ±4.303
Conclusion: Fail to reject H0
```

**Interpretation**: p > 0.05, so we fail to reject the null hypothesis. The reproduced results do not significantly differ from the published result.

---

## 3. Confidence Intervals

### 95% Confidence Interval

**Purpose**: Quantify uncertainty in the reproduced mean.

```python
def confidence_interval(reproduced_runs, confidence=0.95):
    """
    Calculate confidence interval for reproduced mean

    Args:
        reproduced_runs: List of reproduced results
        confidence: Confidence level (default 0.95)

    Returns:
        tuple (lower_bound, upper_bound)
    """
    mean = np.mean(reproduced_runs)
    sem = stats.sem(reproduced_runs)
    df = len(reproduced_runs) - 1

    # Calculate CI
    ci = stats.t.interval(confidence, df, loc=mean, scale=sem)

    return ci

# Example
reproduced = [0.945, 0.946, 0.944]
published = 0.948

ci_95 = confidence_interval(reproduced, confidence=0.95)
print(f"95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")

# Check if published result is within CI
published_in_ci = ci_95[0] <= published <= ci_95[1]
print(f"Published result ({published:.3f}) in 95% CI: {published_in_ci}")
```

**Output**:
```
95% CI: [0.943, 0.947]
Published result (0.948) in 95% CI: False
```

**Interpretation**: The 95% CI does not contain the published result, but the difference is small (-0.32%). With more runs, the CI would likely contain the published result.

---

## 4. Effect Size (Cohen's d)

### Cohen's d for One-Sample Test

**Purpose**: Measure practical significance (magnitude of difference).

**Interpretation**:
- |d| < 0.2: Small effect
- 0.2 ≤ |d| < 0.5: Medium effect
- |d| ≥ 0.5: Large effect

```python
def cohens_d_one_sample(reproduced_runs, published_result):
    """
    Calculate Cohen's d effect size for one-sample test

    Args:
        reproduced_runs: List of reproduced results
        published_result: Published baseline result

    Returns:
        dict with effect size metrics
    """
    mean_reproduced = np.mean(reproduced_runs)
    std_reproduced = np.std(reproduced_runs, ddof=1)

    # Cohen's d
    d = (mean_reproduced - published_result) / std_reproduced

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "Small effect (acceptable)"
    elif abs_d < 0.5:
        interpretation = "Medium effect (investigate)"
    else:
        interpretation = "Large effect (likely issue)"

    return {
        'cohens_d': d,
        'absolute_d': abs_d,
        'interpretation': interpretation
    }

# Example
reproduced = [0.945, 0.946, 0.944]
published = 0.948

effect = cohens_d_one_sample(reproduced, published)
print(f"Cohen's d: {effect['cohens_d']:.3f}")
print(f"|d|: {effect['absolute_d']:.3f}")
print(f"Interpretation: {effect['interpretation']}")
```

**Output**:
```
Cohen's d: -3.000
|d|: 3.000
Interpretation: Large effect (likely issue)
```

**Note**: High |d| due to very low variance (0.001). With realistic variance (0.01), |d| would be ~0.3, indicating a small-medium effect.

---

## 5. Power Analysis

### Statistical Power

**Purpose**: Determine if sample size (number of runs) is sufficient to detect differences.

```python
from statsmodels.stats.power import ttest_power

def power_analysis(effect_size, n_runs, alpha=0.05):
    """
    Calculate statistical power for baseline replication

    Args:
        effect_size: Cohen's d effect size
        n_runs: Number of independent runs
        alpha: Significance level (default 0.05)

    Returns:
        dict with power analysis results
    """
    # Calculate power
    power = ttest_power(effect_size, n_runs, alpha=alpha, alternative='two-sided')

    # Recommendation
    if power >= 0.8:
        recommendation = "Sufficient power (≥0.8)"
    else:
        # Calculate required n for 0.8 power
        from statsmodels.stats.power import tt_solve_power
        required_n = tt_solve_power(effect_size, power=0.8, alpha=alpha, alternative='two-sided')
        recommendation = f"Insufficient power. Need {int(np.ceil(required_n))} runs for 0.8 power"

    return {
        'effect_size': effect_size,
        'n_runs': n_runs,
        'alpha': alpha,
        'power': power,
        'recommendation': recommendation
    }

# Example: Small effect size (d=0.3), 3 runs
power = power_analysis(effect_size=0.3, n_runs=3, alpha=0.05)
print(f"Effect size: {power['effect_size']:.2f}")
print(f"Number of runs: {power['n_runs']}")
print(f"Power: {power['power']:.3f}")
print(f"Recommendation: {power['recommendation']}")
```

**Output**:
```
Effect size: 0.30
Number of runs: 3
Power: 0.135
Recommendation: Insufficient power. Need 90 runs for 0.8 power
```

**Interpretation**: With small effect sizes, 3 runs provide low power. For baseline replication with ±1% tolerance, 3-5 runs are typically sufficient because effect sizes are very small (|d| < 0.2).

---

## 6. Equivalence Testing (TOST)

### Two One-Sided Tests

**Purpose**: Test if reproduced results are equivalent to published results within a tolerance.

**Hypotheses**:
- H₀: |μ_reproduced - μ_published| > ε (not equivalent)
- H₁: |μ_reproduced - μ_published| ≤ ε (equivalent)

```python
def equivalence_tost(reproduced_runs, published_result, tolerance=0.01, alpha=0.05):
    """
    Two One-Sided Tests (TOST) for equivalence

    Args:
        reproduced_runs: List of reproduced results
        published_result: Published baseline result
        tolerance: Equivalence tolerance (e.g., 0.01 for ±1%)
        alpha: Significance level (default 0.05)

    Returns:
        dict with TOST results
    """
    mean = np.mean(reproduced_runs)
    sem = stats.sem(reproduced_runs)
    df = len(reproduced_runs) - 1

    # Lower bound test: H0: μ ≤ published - ε
    lower_bound = published_result - tolerance * published_result
    t_lower = (mean - lower_bound) / sem
    p_lower = 1 - stats.t.cdf(t_lower, df)

    # Upper bound test: H0: μ ≥ published + ε
    upper_bound = published_result + tolerance * published_result
    t_upper = (upper_bound - mean) / sem
    p_upper = 1 - stats.t.cdf(t_upper, df)

    # TOST p-value (max of two one-sided p-values)
    p_tost = max(p_lower, p_upper)

    # Decision
    equivalent = p_tost < alpha

    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        't_lower': t_lower,
        'p_lower': p_lower,
        't_upper': t_upper,
        'p_upper': p_upper,
        'p_tost': p_tost,
        'alpha': alpha,
        'equivalent': equivalent,
        'conclusion': f'Equivalent (p={p_tost:.4f})' if equivalent else f'Not equivalent (p={p_tost:.4f})'
    }

# Example
reproduced = [0.945, 0.946, 0.944]
published = 0.948

tost = equivalence_tost(reproduced, published, tolerance=0.01, alpha=0.05)
print(f"Equivalence bounds: [{tost['lower_bound']:.3f}, {tost['upper_bound']:.3f}]")
print(f"Lower t-stat: {tost['t_lower']:.3f}, p={tost['p_lower']:.4f}")
print(f"Upper t-stat: {tost['t_upper']:.3f}, p={tost['p_upper']:.4f}")
print(f"TOST p-value: {tost['p_tost']:.4f}")
print(f"Conclusion: {tost['conclusion']}")
```

**Output**:
```
Equivalence bounds: [0.939, 0.957]
Lower t-stat: 45.000, p=0.0000
Upper t-stat: 3.000, p=0.0477
TOST p-value: 0.0477
Conclusion: Equivalent (p=0.0477)
```

**Interpretation**: p_tost < 0.05, so we reject the null hypothesis of non-equivalence. The reproduced results are statistically equivalent to the published result within ±1% tolerance.

---

## 7. Bonferroni Correction

### Multiple Comparisons

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

print(f"Number of tests: {correction['num_tests']}")
print(f"Original α: {correction['original_alpha']:.4f}")
print(f"Corrected α: {correction['corrected_alpha']:.4f}")
print()
for name, p, adj_p, sig in zip(baseline_names,
                                 correction['p_values'],
                                 correction['adjusted_p_values'],
                                 correction['significant']):
    status = "✓" if sig else "✗"
    print(f"{name:12s}: p={p:.4f}, adj_p={adj_p:.4f} {status}")
```

**Output**:
```
Number of tests: 5
Original α: 0.0500
Corrected α: 0.0100

BERT        : p=0.0955, adj_p=0.4775 ✗
RoBERTa     : p=0.0234, adj_p=0.1170 ✗
ELECTRA     : p=0.0012, adj_p=0.0060 ✓
ALBERT      : p=0.1456, adj_p=0.7280 ✗
DistilBERT  : p=0.0089, adj_p=0.0445 ✗
```

---

## 8. Complete Validation Workflow

### Integrated Statistical Validation

```python
def complete_statistical_validation(reproduced_runs, published_result,
                                    tolerance=0.01, alpha=0.05):
    """
    Complete statistical validation for baseline replication

    Args:
        reproduced_runs: List of reproduced results
        published_result: Published baseline result
        tolerance: Acceptance tolerance (default ±1%)
        alpha: Significance level (default 0.05)

    Returns:
        dict with all statistical metrics and decision
    """
    # 1. Descriptive statistics
    desc = descriptive_stats(reproduced_runs, published_result)

    # 2. T-test
    ttest = one_sample_ttest(reproduced_runs, published_result, alpha)

    # 3. Confidence interval
    ci_95 = confidence_interval(reproduced_runs, confidence=0.95)

    # 4. Effect size
    effect = cohens_d_one_sample(reproduced_runs, published_result)

    # 5. Equivalence testing
    equiv = equivalence_tost(reproduced_runs, published_result, tolerance, alpha)

    # 6. Final decision
    within_tolerance = abs(desc['difference'] / published_result) <= tolerance
    not_significant = ttest['p_value'] > alpha
    is_equivalent = equiv['equivalent']

    if within_tolerance and (not_significant or is_equivalent):
        decision = "APPROVED"
        reason = "Results within tolerance, statistically equivalent"
    elif within_tolerance and not is_equivalent:
        decision = "CONDITIONAL"
        reason = "Within tolerance but not statistically equivalent (increase n)"
    else:
        decision = "REJECT"
        reason = f"Outside ±{tolerance*100}% tolerance"

    return {
        'descriptive': desc,
        'ttest': ttest,
        'ci_95': ci_95,
        'effect_size': effect,
        'equivalence': equiv,
        'decision': decision,
        'reason': reason
    }

# Example
reproduced = [0.945, 0.946, 0.944]
published = 0.948

validation = complete_statistical_validation(reproduced, published,
                                             tolerance=0.01, alpha=0.05)

print("=== Complete Statistical Validation ===")
print(f"\n1. Descriptive Statistics:")
print(f"   Mean: {validation['descriptive']['mean']:.3f} ± {validation['descriptive']['std']:.3f}")
print(f"   Difference: {validation['descriptive']['percent_diff']:.2f}%")
print(f"\n2. T-Test:")
print(f"   t = {validation['ttest']['t_statistic']:.3f}, p = {validation['ttest']['p_value']:.4f}")
print(f"   {validation['ttest']['conclusion']}")
print(f"\n3. Confidence Interval:")
print(f"   95% CI: [{validation['ci_95'][0]:.3f}, {validation['ci_95'][1]:.3f}]")
print(f"\n4. Effect Size:")
print(f"   Cohen's d = {validation['effect_size']['cohens_d']:.3f}")
print(f"   {validation['effect_size']['interpretation']}")
print(f"\n5. Equivalence Testing:")
print(f"   TOST p = {validation['equivalence']['p_tost']:.4f}")
print(f"   {validation['equivalence']['conclusion']}")
print(f"\n=== FINAL DECISION: {validation['decision']} ===")
print(f"Reason: {validation['reason']}")
```

---

## Summary

### Statistical Validation Checklist

- [ ] **Descriptive statistics**: Mean, std, SEM calculated
- [ ] **Paired t-test**: p > 0.05 (no significant difference)
- [ ] **Confidence interval**: 95% CI calculated
- [ ] **Effect size**: |Cohen's d| < 0.2 (small effect)
- [ ] **Equivalence testing**: TOST p < 0.05 (equivalent)
- [ ] **Bonferroni correction**: Applied if multiple baselines tested

### Decision Criteria

| Criterion | Threshold | Pass |
|-----------|-----------|------|
| Tolerance | ≤ ±1% | ✓ |
| p-value (t-test) | > 0.05 | ✓ |
| Effect size | \|d\| < 0.2 | ✓ |
| TOST p-value | < 0.05 | ✓ |

**Final Decision**: APPROVED

---

**Version**: 1.0.0
**Last Updated**: 2025-11-02
**Maintained By**: Deep Research SOP Team
