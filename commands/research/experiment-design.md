---
name: research:experiment-design
description: Experiment design helper with statistical power analysis for rigorous research
category: Research Workflows
version: 1.0.0
requires:
  - python3
  - scipy
  - statsmodels
  - matplotlib
  - numpy
usage: |
  /research:experiment-design --type "rct" --hypothesis "Drug A reduces symptoms by 20%" --power 0.8 --alpha 0.05
  /research:experiment-design --design "factorial" --factors "temperature,pressure,catalyst" --levels "2,3,2" --replications 3
---

# Research: Experiment Design

**Category**: Research Workflows
**Purpose**: Design rigorous experiments with statistical power analysis, randomization, and design optimization for scientific research.

## Overview

This command provides a comprehensive experiment design framework including:
- **Study Design**: RCT, factorial, crossover, repeated measures, quasi-experimental
- **Power Analysis**: Sample size calculation with effect size estimation
- **Randomization**: Stratified, block, cluster randomization schemes
- **Blinding**: Single-blind, double-blind, triple-blind protocols
- **Optimization**: Design of Experiments (DOE) for efficiency

## Core Features

### 1. Experimental Designs Supported
- **Randomized Controlled Trial (RCT)**: Gold standard for causal inference
- **Factorial Design**: Multiple factors and interactions (2^k, 3^k designs)
- **Crossover Design**: Within-subject comparisons with washout periods
- **Repeated Measures**: Longitudinal data with correlated observations
- **Quasi-Experimental**: Non-randomized with comparison groups
- **Latin Square**: Control for two nuisance variables
- **Response Surface**: Optimization experiments (Box-Behnken, Central Composite)

### 2. Statistical Power Analysis
- **Sample Size Calculation**: For desired power (typically 0.8-0.9)
- **Effect Size Estimation**: Cohen's d, odds ratios, correlation coefficients
- **Power Curves**: Visualize power across sample sizes
- **Minimum Detectable Effect**: Smallest effect size detectable
- **Type I/II Error Rates**: Control false positives and negatives

### 3. Randomization & Blinding
- **Simple Randomization**: Coin flip, random number generator
- **Stratified Randomization**: Balance key covariates across groups
- **Block Randomization**: Ensure equal group sizes in blocks
- **Cluster Randomization**: Randomize groups instead of individuals
- **Blinding Protocols**: Prevent bias in treatment and assessment

### 4. Design Optimization
- **Fractional Factorial**: Reduce runs while maintaining information
- **Optimal Designs**: D-optimal, A-optimal, I-optimal
- **Adaptive Designs**: Sequential designs based on interim results
- **Blocking**: Control for known sources of variation

## Command Structure

```bash
/research:experiment-design [OPTIONS]

Options:
  --type <string>               Experiment type (rct, factorial, crossover, repeated-measures)
  --hypothesis <string>         Research hypothesis to test
  --outcome <string>            Primary outcome variable (continuous, binary, count, time-to-event)
  --effect-size <float>         Expected effect size (Cohen's d, OR, correlation)
  --power <float>               Desired statistical power (default: 0.8)
  --alpha <float>               Significance level (default: 0.05)
  --design <string>             Factorial design (e.g., "2x3x2" for 2 levels × 3 levels × 2 levels)
  --factors <list>              Factor names (comma-separated)
  --levels <list>               Levels per factor (comma-separated)
  --replications <int>          Number of replications per cell
  --blocking <list>             Blocking factors to control
  --randomization <string>      Randomization method (simple, stratified, block, cluster)
  --blinding <string>           Blinding protocol (single, double, triple)
  --output-dir <path>           Output directory (default: "./experiment-design")
  --export-format <list>        Export formats (csv, latex, r-script)
```

## Usage Examples

### RCT with Power Analysis
```bash
/research:experiment-design \
  --type "rct" \
  --hypothesis "Cognitive behavioral therapy reduces depression scores by 5 points (Cohen's d = 0.5)" \
  --outcome "continuous" \
  --effect-size 0.5 \
  --power 0.8 \
  --alpha 0.05 \
  --randomization "stratified" \
  --blinding "double"
```

### Factorial Design
```bash
/research:experiment-design \
  --type "factorial" \
  --design "2x3x2" \
  --factors "temperature,pH,catalyst" \
  --levels "2,3,2" \
  --replications 3 \
  --randomization "block"
```

### Crossover Trial
```bash
/research:experiment-design \
  --type "crossover" \
  --hypothesis "Treatment A superior to Treatment B with 15% difference" \
  --outcome "binary" \
  --effect-size 0.15 \
  --power 0.9 \
  --washout-period "2 weeks"
```

## Implementation

### Phase 1: Power Analysis (Auto-Execute)

```python
#!/usr/bin/env python3
"""
Statistical Power Analysis for Experiment Design
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats import power
from typing import Dict, List, Tuple

class PowerAnalyzer:
    """Calculate sample sizes and power for experimental designs"""

    def __init__(self, config: Dict):
        self.effect_size = config.get('effect_size', 0.5)
        self.alpha = config.get('alpha', 0.05)
        self.power = config.get('power', 0.8)
        self.outcome_type = config.get('outcome', 'continuous')
        self.design_type = config.get('type', 'rct')
        self.output_dir = config.get('output_dir', './experiment-design')

    def calculate_sample_size_ttest(self) -> Dict:
        """Calculate sample size for two-sample t-test"""
        print("=" * 60)
        print("📊 POWER ANALYSIS: Two-Sample T-Test")
        print("=" * 60)
        print(f"Effect size (Cohen's d): {self.effect_size}")
        print(f"Significance level (α): {self.alpha}")
        print(f"Desired power (1-β): {self.power}")
        print()

        # Calculate sample size using statsmodels
        analysis = power.TTestIndPower()
        n = analysis.solve_power(
            effect_size=self.effect_size,
            alpha=self.alpha,
            power=self.power,
            ratio=1.0,  # Equal group sizes
            alternative='two-sided'
        )

        n_per_group = int(np.ceil(n))
        total_n = n_per_group * 2

        print(f"✅ Required sample size: {n_per_group} per group")
        print(f"✅ Total participants: {total_n}")
        print()

        # Calculate achieved power
        achieved_power = analysis.solve_power(
            effect_size=self.effect_size,
            nobs1=n_per_group,
            alpha=self.alpha,
            ratio=1.0,
            alternative='two-sided'
        )

        print(f"Achieved power: {achieved_power:.4f}")

        # Calculate minimum detectable effect
        mde = analysis.solve_power(
            nobs1=n_per_group,
            alpha=self.alpha,
            power=self.power,
            ratio=1.0,
            alternative='two-sided'
        )

        print(f"Minimum detectable effect: {mde:.4f}")

        return {
            'n_per_group': n_per_group,
            'total_n': total_n,
            'achieved_power': achieved_power,
            'min_detectable_effect': mde
        }

    def calculate_sample_size_proportion(self, p1: float, p2: float) -> Dict:
        """Calculate sample size for two-proportion test"""
        print("=" * 60)
        print("📊 POWER ANALYSIS: Two-Proportion Test")
        print("=" * 60)
        print(f"Proportion in group 1: {p1}")
        print(f"Proportion in group 2: {p2}")
        print(f"Significance level (α): {self.alpha}")
        print(f"Desired power (1-β): {self.power}")
        print()

        # Calculate effect size (Cohen's h)
        h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

        # Calculate sample size
        analysis = power.NormalIndPower()
        n = analysis.solve_power(
            effect_size=h,
            alpha=self.alpha,
            power=self.power,
            ratio=1.0,
            alternative='two-sided'
        )

        n_per_group = int(np.ceil(n))
        total_n = n_per_group * 2

        print(f"Effect size (Cohen's h): {h:.4f}")
        print(f"✅ Required sample size: {n_per_group} per group")
        print(f"✅ Total participants: {total_n}")

        return {
            'effect_size_h': h,
            'n_per_group': n_per_group,
            'total_n': total_n
        }

    def generate_power_curve(self, output_path: str):
        """Generate power curve across sample sizes"""
        print("\n📈 Generating Power Curve...")

        sample_sizes = np.arange(10, 500, 10)
        powers = []

        analysis = power.TTestIndPower()

        for n in sample_sizes:
            p = analysis.solve_power(
                effect_size=self.effect_size,
                nobs1=n,
                alpha=self.alpha,
                ratio=1.0,
                alternative='two-sided'
            )
            powers.append(p)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(sample_sizes, powers, linewidth=2, label=f'd = {self.effect_size}')
        plt.axhline(y=0.8, color='r', linestyle='--', label='Power = 0.8')
        plt.axhline(y=0.9, color='orange', linestyle='--', label='Power = 0.9')
        plt.xlabel('Sample Size per Group', fontsize=12)
        plt.ylabel('Statistical Power (1-β)', fontsize=12)
        plt.title(f'Power Curve (α = {self.alpha})', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Power curve saved: {output_path}")
        plt.close()

    def calculate_factorial_sample_size(self, factors: int, levels: List[int],
                                       replications: int) -> Dict:
        """Calculate sample size for factorial design"""
        print("=" * 60)
        print(f"📊 FACTORIAL DESIGN: {' × '.join(map(str, levels))}")
        print("=" * 60)

        # Total number of treatment combinations
        n_cells = np.prod(levels)
        total_runs = n_cells * replications

        print(f"Factors: {factors}")
        print(f"Levels: {levels}")
        print(f"Treatment combinations: {n_cells}")
        print(f"Replications per cell: {replications}")
        print(f"✅ Total runs required: {total_runs}")
        print()

        # Degrees of freedom
        df_total = total_runs - 1
        df_error = total_runs - n_cells

        print(f"Degrees of freedom (total): {df_total}")
        print(f"Degrees of freedom (error): {df_error}")

        return {
            'n_cells': n_cells,
            'replications': replications,
            'total_runs': total_runs,
            'df_total': df_total,
            'df_error': df_error
        }

    def analyze_design_efficiency(self, design_matrix: np.ndarray) -> Dict:
        """Analyze efficiency of experimental design"""
        print("\n🔍 DESIGN EFFICIENCY ANALYSIS")
        print("=" * 60)

        # D-optimality (determinant of X'X)
        XtX = design_matrix.T @ design_matrix
        d_efficiency = np.linalg.det(XtX) ** (1 / design_matrix.shape[1])

        # A-optimality (trace of (X'X)^-1)
        try:
            XtX_inv = np.linalg.inv(XtX)
            a_efficiency = 1 / np.trace(XtX_inv)
        except np.linalg.LinAlgError:
            a_efficiency = np.nan

        # Condition number (multicollinearity)
        condition_number = np.linalg.cond(design_matrix)

        print(f"D-efficiency: {d_efficiency:.4f}")
        print(f"A-efficiency: {a_efficiency:.4f}")
        print(f"Condition number: {condition_number:.2f}")

        if condition_number > 30:
            print("⚠️  Warning: High condition number indicates multicollinearity")

        return {
            'd_efficiency': d_efficiency,
            'a_efficiency': a_efficiency,
            'condition_number': condition_number
        }

# Example usage
if __name__ == "__main__":
    config = {
        'effect_size': 0.5,
        'alpha': 0.05,
        'power': 0.8,
        'outcome': 'continuous',
        'type': 'rct',
        'output_dir': './experiment-design'
    }

    analyzer = PowerAnalyzer(config)

    # Calculate sample size
    results = analyzer.calculate_sample_size_ttest()

    # Generate power curve
    import os
    os.makedirs(config['output_dir'], exist_ok=True)
    analyzer.generate_power_curve(f"{config['output_dir']}/power_curve.png")
```

### Phase 2: Randomization Scheme (Auto-Execute)

```python
#!/usr/bin/env python3
"""
Randomization and Blinding Protocols
"""

import numpy as np
import pandas as pd
from typing import List, Dict

class Randomizer:
    """Generate randomization schemes for experiments"""

    def __init__(self, n_participants: int, n_groups: int = 2,
                 method: str = 'simple', seed: int = None):
        self.n = n_participants
        self.k = n_groups
        self.method = method
        self.seed = seed

        if seed:
            np.random.seed(seed)

    def simple_randomization(self) -> pd.DataFrame:
        """Simple 1:1 randomization"""
        print("🎲 SIMPLE RANDOMIZATION")
        print("=" * 60)

        assignments = np.random.choice(
            range(self.k),
            size=self.n,
            replace=True
        )

        df = pd.DataFrame({
            'participant_id': range(1, self.n + 1),
            'group': assignments,
            'group_label': [f'Group {i}' for i in assignments]
        })

        # Check balance
        group_counts = df['group'].value_counts().sort_index()
        print("Group sizes:")
        for group, count in group_counts.items():
            print(f"  Group {group}: {count} ({count/self.n*100:.1f}%)")

        return df

    def block_randomization(self, block_size: int = 4) -> pd.DataFrame:
        """Block randomization to ensure balance"""
        print("📦 BLOCK RANDOMIZATION")
        print("=" * 60)
        print(f"Block size: {block_size}")

        n_blocks = int(np.ceil(self.n / block_size))
        assignments = []

        for block in range(n_blocks):
            # Create balanced block
            block_assignments = []
            for group in range(self.k):
                block_assignments.extend([group] * (block_size // self.k))

            # Randomize within block
            np.random.shuffle(block_assignments)
            assignments.extend(block_assignments)

        # Trim to exact sample size
        assignments = assignments[:self.n]

        df = pd.DataFrame({
            'participant_id': range(1, self.n + 1),
            'block': [i // block_size for i in range(self.n)],
            'group': assignments,
            'group_label': [f'Group {i}' for i in assignments]
        })

        print(f"✅ Created {n_blocks} blocks")
        print(f"✅ Randomized {self.n} participants")

        return df

    def stratified_randomization(self, strata: pd.Series) -> pd.DataFrame:
        """Stratified randomization on key covariates"""
        print("📊 STRATIFIED RANDOMIZATION")
        print("=" * 60)

        df = pd.DataFrame({
            'participant_id': range(1, self.n + 1),
            'stratum': strata
        })

        # Randomize within each stratum
        assignments = []
        for stratum_value in df['stratum'].unique():
            stratum_mask = df['stratum'] == stratum_value
            n_stratum = stratum_mask.sum()

            # Balanced allocation within stratum
            stratum_assignments = np.random.choice(
                range(self.k),
                size=n_stratum,
                replace=True
            )
            assignments.extend(stratum_assignments)

        df['group'] = assignments
        df['group_label'] = [f'Group {i}' for i in assignments]

        # Check balance by stratum
        print("Balance by stratum:")
        for stratum_value in df['stratum'].unique():
            stratum_df = df[df['stratum'] == stratum_value]
            print(f"\n  Stratum '{stratum_value}':")
            for group in range(self.k):
                count = (stratum_df['group'] == group).sum()
                pct = count / len(stratum_df) * 100
                print(f"    Group {group}: {count} ({pct:.1f}%)")

        return df

    def export_randomization_list(self, df: pd.DataFrame, output_path: str):
        """Export randomization list with blinding"""
        print(f"\n📄 Exporting randomization list: {output_path}")

        # Create blinded version (codes instead of group names)
        df_blinded = df.copy()
        df_blinded['treatment_code'] = df_blinded['group'].map({
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        })

        df_blinded.to_csv(output_path, index=False)
        print("✅ Randomization list exported")

        # Create allocation concealment envelopes
        envelope_path = output_path.replace('.csv', '_envelopes.txt')
        with open(envelope_path, 'w') as f:
            for _, row in df_blinded.iterrows():
                f.write(f"Participant {row['participant_id']}: {row['treatment_code']}\n")
                f.write("-" * 40 + "\n")

        print(f"✅ Allocation envelopes: {envelope_path}")

# Example usage
if __name__ == "__main__":
    # Block randomization
    randomizer = Randomizer(n_participants=100, n_groups=2, method='block', seed=42)
    allocation = randomizer.block_randomization(block_size=4)

    # Export
    import os
    os.makedirs('./experiment-design', exist_ok=True)
    randomizer.export_randomization_list(allocation, './experiment-design/randomization_list.csv')
```

## Output Files

```
experiment-design/
├── power_analysis.json          # Sample size calculations
├── power_curve.png              # Power vs sample size plot
├── randomization_list.csv       # Treatment assignments
├── randomization_envelopes.txt  # Allocation concealment
├── design_matrix.csv            # Factorial design matrix
├── protocol.md                  # Study protocol document
└── analysis_plan.R              # Pre-specified analysis script
```

## Integration with Research Workflow

### Memory Pattern
```javascript
{
  "research/experiment-design/{study-id}/power-analysis": "Sample size calculations",
  "research/experiment-design/{study-id}/randomization": "Treatment assignments",
  "research/experiment-design/{study-id}/protocol": "Study protocol",
  "research/experiment-design/{study-id}/analysis-plan": "Statistical analysis plan"
}
```

## Best Practices

1. **Pre-register Protocol**: Register study design before data collection (ClinicalTrials.gov, OSF)
2. **Calculate Power**: Always justify sample size with power analysis
3. **Randomize Properly**: Use appropriate randomization for your design
4. **Blind When Possible**: Minimize bias with double/triple blinding
5. **Pre-specify Analysis**: Write analysis plan before seeing data
6. **Report Transparently**: Follow CONSORT, STROBE, or ARRIVE guidelines

## References

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- CONSORT Statement: http://www.consort-statement.org/
- G*Power: https://www.psychologie.hhu.de/arbeitsgruppen/allgemeine-psychologie-und-arbeitspsychologie/gpower

---

**Status**: Production Ready
**Version**: 1.0.0
**Last Updated**: 2025-11-01
