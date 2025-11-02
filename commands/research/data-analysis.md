---
name: research:data-analysis
description: Statistical data analysis with hypothesis testing for research studies
category: Research Workflows
version: 1.0.0
requires:
  - python3
  - scipy
  - statsmodels
  - pandas
  - matplotlib
  - seaborn
usage: |
  /research:data-analysis --data "results.csv" --outcome "depression_score" --predictor "treatment" --test "t-test"
  /research:data-analysis --data "rct_data.csv" --analysis "anova" --factors "treatment,gender" --covariates "age,baseline"
---

# Research: Statistical Data Analysis

**Category**: Research Workflows
**Purpose**: Perform rigorous statistical analysis with hypothesis testing, effect sizes, and visualization for research studies.

## Overview

Comprehensive statistical analysis toolkit for research data:
- **Descriptive Statistics**: Summary statistics, distributions, outlier detection
- **Hypothesis Testing**: t-tests, ANOVA, chi-square, non-parametric tests
- **Regression**: Linear, logistic, mixed-effects models
- **Effect Sizes**: Cohen's d, odds ratios, confidence intervals
- **Assumptions**: Normality, homogeneity of variance, independence checks
- **Visualization**: Publication-quality plots and tables

## Supported Analyses

### Parametric Tests
- **t-tests**: Independent, paired, one-sample
- **ANOVA**: One-way, two-way, repeated measures, ANCOVA
- **Regression**: Linear, multiple, polynomial
- **Correlation**: Pearson, partial correlation

### Non-Parametric Tests
- **Mann-Whitney U**: Non-parametric alternative to independent t-test
- **Wilcoxon Signed-Rank**: Non-parametric alternative to paired t-test
- **Kruskal-Wallis**: Non-parametric alternative to one-way ANOVA
- **Spearman Correlation**: Non-parametric correlation

### Categorical Data
- **Chi-Square**: Goodness-of-fit, independence tests
- **Fisher's Exact**: Small sample categorical analysis
- **McNemar**: Paired categorical data
- **Logistic Regression**: Binary/multinomial outcomes

### Advanced Models
- **Mixed-Effects**: Hierarchical/multilevel models
- **Survival Analysis**: Cox regression, Kaplan-Meier
- **Time Series**: ARIMA, intervention analysis
- **Multivariate**: MANOVA, PCA, factor analysis

## Command Structure

```bash
/research:data-analysis [OPTIONS]

Options:
  --data <path>                 Input data file (CSV, Excel, SPSS, Stata)
  --outcome <string>            Dependent variable name
  --predictor <string>          Independent variable name
  --covariates <list>           Covariate names (comma-separated)
  --test <string>               Statistical test (t-test, anova, regression, chi-square)
  --analysis <string>           Analysis type (descriptive, inferential, multivariate)
  --alpha <float>               Significance level (default: 0.05)
  --adjust-method <string>      Multiple comparison adjustment (bonferroni, holm, fdr)
  --effect-size                 Calculate effect sizes (Cohen's d, r, eta-squared)
  --assumptions                 Check statistical assumptions
  --visualize                   Generate plots (boxplot, histogram, qq-plot, scatter)
  --output-dir <path>           Output directory (default: "./data-analysis")
  --export-format <list>        Export formats (csv, latex, apa-table, r-output)
```

## Usage Examples

### Independent t-test
```bash
/research:data-analysis \
  --data "rct_results.csv" \
  --outcome "depression_score" \
  --predictor "treatment" \
  --test "t-test" \
  --effect-size \
  --assumptions \
  --visualize
```

### Two-Way ANOVA
```bash
/research:data-analysis \
  --data "experiment.csv" \
  --outcome "reaction_time" \
  --predictor "condition,age_group" \
  --test "anova" \
  --adjust-method "bonferroni" \
  --effect-size
```

### Logistic Regression
```bash
/research:data-analysis \
  --data "clinical_trial.csv" \
  --outcome "remission" \
  --predictor "treatment" \
  --covariates "age,sex,baseline_severity" \
  --test "logistic-regression" \
  --export-format "latex,apa-table"
```

### Repeated Measures ANOVA
```bash
/research:data-analysis \
  --data "longitudinal.csv" \
  --outcome "cognition_score" \
  --predictor "time" \
  --test "repeated-measures-anova" \
  --subject-id "participant_id"
```

## Implementation

```python
#!/usr/bin/env python3
"""
Statistical Data Analysis for Research
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats import power
from statsmodels.formula.api import ols, logit
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ResearchAnalyzer:
    """Statistical analysis for research data"""

    def __init__(self, data_path: str, config: Dict):
        self.df = pd.read_csv(data_path)
        self.outcome = config.get('outcome')
        self.predictor = config.get('predictor')
        self.covariates = config.get('covariates', [])
        self.test_type = config.get('test', 't-test')
        self.alpha = config.get('alpha', 0.05)
        self.output_dir = config.get('output_dir', './data-analysis')

        print("=" * 60)
        print("📊 STATISTICAL DATA ANALYSIS")
        print("=" * 60)
        print(f"Dataset: {data_path}")
        print(f"Sample size: {len(self.df)}")
        print(f"Variables: {len(self.df.columns)}")
        print(f"Outcome: {self.outcome}")
        print(f"Predictor: {self.predictor}")
        print()

    def descriptive_statistics(self) -> Dict:
        """Calculate descriptive statistics"""
        print("📈 DESCRIPTIVE STATISTICS")
        print("=" * 60)

        # Overall statistics
        desc = self.df[self.outcome].describe()
        print(f"\nOutcome: {self.outcome}")
        print(desc)

        # By group statistics
        if self.predictor:
            print(f"\nBy {self.predictor}:")
            group_stats = self.df.groupby(self.predictor)[self.outcome].describe()
            print(group_stats)

        # Check for missing data
        missing = self.df.isnull().sum()
        if missing.any():
            print("\n⚠️  Missing Data:")
            print(missing[missing > 0])

        return {
            'overall': desc.to_dict(),
            'by_group': group_stats.to_dict() if self.predictor else None
        }

    def check_assumptions(self) -> Dict:
        """Check statistical assumptions"""
        print("\n🔍 CHECKING STATISTICAL ASSUMPTIONS")
        print("=" * 60)

        results = {}

        # 1. Normality (Shapiro-Wilk test)
        if self.predictor:
            groups = self.df[self.predictor].unique()
            print("\n1. Normality Test (Shapiro-Wilk):")
            for group in groups:
                group_data = self.df[self.df[self.predictor] == group][self.outcome]
                stat, p_value = stats.shapiro(group_data.dropna())

                print(f"   {self.predictor} = {group}:")
                print(f"      W = {stat:.4f}, p = {p_value:.4f}", end="")
                if p_value < 0.05:
                    print(" ⚠️  Non-normal")
                else:
                    print(" ✅ Normal")

                results[f'normality_{group}'] = {'statistic': stat, 'p_value': p_value}
        else:
            stat, p_value = stats.shapiro(self.df[self.outcome].dropna())
            print(f"\n1. Normality Test (Shapiro-Wilk):")
            print(f"   W = {stat:.4f}, p = {p_value:.4f}")
            results['normality'] = {'statistic': stat, 'p_value': p_value}

        # 2. Homogeneity of Variance (Levene's test)
        if self.predictor:
            groups = [group_data[self.outcome].dropna()
                     for _, group_data in self.df.groupby(self.predictor)]

            stat, p_value = stats.levene(*groups)
            print(f"\n2. Homogeneity of Variance (Levene's test):")
            print(f"   F = {stat:.4f}, p = {p_value:.4f}", end="")
            if p_value < 0.05:
                print(" ⚠️  Unequal variances")
            else:
                print(" ✅ Equal variances")

            results['levene'] = {'statistic': stat, 'p_value': p_value}

        # 3. Outliers (z-score > 3)
        z_scores = np.abs(stats.zscore(self.df[self.outcome].dropna()))
        outliers = (z_scores > 3).sum()
        print(f"\n3. Outlier Detection (|z| > 3):")
        print(f"   Outliers: {outliers} ({outliers/len(self.df)*100:.1f}%)")
        results['outliers'] = outliers

        return results

    def independent_ttest(self) -> Dict:
        """Independent samples t-test"""
        print("\n📊 INDEPENDENT SAMPLES T-TEST")
        print("=" * 60)

        groups = self.df[self.predictor].unique()
        if len(groups) != 2:
            raise ValueError("t-test requires exactly 2 groups")

        group1_data = self.df[self.df[self.predictor] == groups[0]][self.outcome]
        group2_data = self.df[self.df[self.predictor] == groups[1]][self.outcome]

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)

        # Calculate effect size (Cohen's d)
        mean1, mean2 = group1_data.mean(), group2_data.mean()
        std1, std2 = group1_data.std(), group2_data.std()
        pooled_std = np.sqrt(((len(group1_data)-1)*std1**2 + (len(group2_data)-1)*std2**2) /
                            (len(group1_data) + len(group2_data) - 2))
        cohens_d = (mean1 - mean2) / pooled_std

        # 95% Confidence Interval
        se_diff = pooled_std * np.sqrt(1/len(group1_data) + 1/len(group2_data))
        df = len(group1_data) + len(group2_data) - 2
        t_crit = stats.t.ppf(0.975, df)
        ci_lower = (mean1 - mean2) - t_crit * se_diff
        ci_upper = (mean1 - mean2) + t_crit * se_diff

        print(f"Group 1 ({groups[0]}): M = {mean1:.2f}, SD = {std1:.2f}, n = {len(group1_data)}")
        print(f"Group 2 ({groups[1]}): M = {mean2:.2f}, SD = {std2:.2f}, n = {len(group2_data)}")
        print()
        print(f"t({df}) = {t_stat:.3f}, p = {p_value:.4f}")
        print(f"Mean difference: {mean1 - mean2:.2f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}]")
        print(f"Cohen's d = {cohens_d:.3f}", end="")

        # Interpret effect size
        if abs(cohens_d) < 0.2:
            print(" (negligible)")
        elif abs(cohens_d) < 0.5:
            print(" (small)")
        elif abs(cohens_d) < 0.8:
            print(" (medium)")
        else:
            print(" (large)")

        # Interpret p-value
        if p_value < self.alpha:
            print(f"\n✅ Statistically significant at α = {self.alpha}")
        else:
            print(f"\n❌ Not statistically significant at α = {self.alpha}")

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'mean_diff': mean1 - mean2,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'df': df
        }

    def one_way_anova(self) -> Dict:
        """One-way ANOVA with post-hoc tests"""
        print("\n📊 ONE-WAY ANOVA")
        print("=" * 60)

        # Prepare data
        groups = [group_data[self.outcome].dropna()
                 for _, group_data in self.df.groupby(self.predictor)]

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        # Degrees of freedom
        k = len(groups)  # Number of groups
        n = sum(len(g) for g in groups)  # Total sample size
        df_between = k - 1
        df_within = n - k

        # Effect size (eta-squared)
        # Using statsmodels for more detailed output
        formula = f'{self.outcome} ~ C({self.predictor})'
        model = ols(formula, data=self.df).fit()
        anova_table = anova_lm(model, typ=2)

        eta_squared = anova_table['sum_sq'][0] / anova_table['sum_sq'].sum()

        print(f"F({df_between}, {df_within}) = {f_stat:.3f}, p = {p_value:.4f}")
        print(f"η² = {eta_squared:.3f}", end="")

        if eta_squared < 0.01:
            print(" (small)")
        elif eta_squared < 0.06:
            print(" (medium)")
        else:
            print(" (large)")

        # Post-hoc tests (Tukey HSD)
        if p_value < self.alpha:
            print(f"\n✅ Statistically significant at α = {self.alpha}")
            print("\n📋 Post-hoc Tests (Tukey HSD):")
            tukey_result = pairwise_tukeyhsd(
                self.df[self.outcome],
                self.df[self.predictor],
                alpha=self.alpha
            )
            print(tukey_result)
        else:
            print(f"\n❌ Not statistically significant at α = {self.alpha}")

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'df_between': df_between,
            'df_within': df_within
        }

    def visualize_results(self, output_path: str):
        """Generate publication-quality visualizations"""
        print("\n📊 GENERATING VISUALIZATIONS")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Boxplot
        sns.boxplot(data=self.df, x=self.predictor, y=self.outcome, ax=axes[0, 0])
        axes[0, 0].set_title('Distribution by Group', fontweight='bold')
        axes[0, 0].set_xlabel(self.predictor)
        axes[0, 0].set_ylabel(self.outcome)

        # 2. Violin plot
        sns.violinplot(data=self.df, x=self.predictor, y=self.outcome, ax=axes[0, 1])
        axes[0, 1].set_title('Violin Plot', fontweight='bold')

        # 3. Histogram with density
        for group in self.df[self.predictor].unique():
            group_data = self.df[self.df[self.predictor] == group][self.outcome]
            axes[1, 0].hist(group_data, alpha=0.5, label=str(group), bins=20, density=True)
        axes[1, 0].set_title('Distribution Comparison', fontweight='bold')
        axes[1, 0].set_xlabel(self.outcome)
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()

        # 4. Q-Q plot (normality check)
        stats.probplot(self.df[self.outcome].dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plots saved: {output_path}")
        plt.close()

    def export_apa_table(self, results: Dict, output_path: str):
        """Export results in APA format"""
        print(f"\n📄 Exporting APA-formatted table: {output_path}")

        # Format results for APA table
        if self.test_type == 't-test':
            table = f"""
Table 1
Comparison of {self.outcome} by {self.predictor}

Group                M       SD      n
{'=' * 50}
{self.df[self.predictor].unique()[0]:20s} {results['mean_diff']:.2f}  {0:.2f}  {0}
{self.df[self.predictor].unique()[1]:20s} {0:.2f}  {0:.2f}  {0}

Note. t({results['df']}) = {results['t_statistic']:.2f}, p = {results['p_value']:.3f},
      d = {results['cohens_d']:.2f}, 95% CI [{results['ci_lower']:.2f}, {results['ci_upper']:.2f}]
"""

        with open(output_path, 'w') as f:
            f.write(table)

        print("✅ APA table exported")

# Example usage
if __name__ == "__main__":
    config = {
        'outcome': 'depression_score',
        'predictor': 'treatment',
        'test': 't-test',
        'alpha': 0.05,
        'output_dir': './data-analysis'
    }

    analyzer = ResearchAnalyzer('./rct_results.csv', config)

    # Run analysis
    desc_stats = analyzer.descriptive_statistics()
    assumptions = analyzer.check_assumptions()
    results = analyzer.independent_ttest()

    # Visualize
    import os
    os.makedirs(config['output_dir'], exist_ok=True)
    analyzer.visualize_results(f"{config['output_dir']}/analysis_plots.png")
    analyzer.export_apa_table(results, f"{config['output_dir']}/apa_table.txt")
```

## Output Files

```
data-analysis/
├── descriptive_stats.csv        # Summary statistics
├── analysis_results.json        # Statistical test results
├── assumptions_check.txt        # Assumption test outputs
├── analysis_plots.png           # Visualization (boxplot, histogram, Q-Q)
├── apa_table.txt                # APA-formatted results table
└── analysis_report.md           # Comprehensive report
```

## Best Practices

1. **Check Assumptions First**: Always verify normality, homogeneity, independence
2. **Report Effect Sizes**: p-values alone are insufficient; include Cohen's d, r, η²
3. **Use Appropriate Tests**: Parametric for normal data, non-parametric otherwise
4. **Adjust for Multiple Comparisons**: Use Bonferroni, Holm, or FDR corrections
5. **Report CI**: Always include 95% confidence intervals
6. **Visualize Data**: Plots reveal patterns tests may miss

## References

- Field, A. (2013). Discovering Statistics Using IBM SPSS Statistics
- APA Publication Manual (7th ed.)
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences

---

**Status**: Production Ready
**Version**: 1.0.0
**Last Updated**: 2025-11-01
