#!/usr/bin/env python3
"""
Loss Curve Analyzer

Analyzes training loss curves to detect anomalies, trends, and pathological behavior.
Provides visualizations and diagnostic insights for ML training debugging.

Usage:
    python loss-analyzer.py --log-file train.log --output loss_analysis.png
    python loss-analyzer.py --csv metrics.csv --detect-divergence --window 5
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter


class LossAnalyzer:
    """Analyzes training loss curves for anomalies and patterns."""

    def __init__(self, window_size: int = 5, divergence_threshold: float = 0.15):
        self.window_size = window_size
        self.divergence_threshold = divergence_threshold
        self.metrics = {}

    def parse_log_file(self, log_path: Path) -> pd.DataFrame:
        """Parse training log file to extract loss metrics."""
        data = []

        # Common log patterns
        patterns = [
            r'Epoch (\d+).*?Loss: ([\d.]+)',
            r'Step (\d+).*?loss=([\d.]+)',
            r'\[(\d+)\].*?train_loss: ([\d.]+)',
        ]

        with open(log_path, 'r') as f:
            for line in f:
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        step, loss = match.groups()
                        data.append({
                            'step': int(step),
                            'loss': float(loss),
                            'raw_line': line.strip()
                        })
                        break

        if not data:
            raise ValueError(f"No loss data found in {log_path}")

        df = pd.DataFrame(data)
        return df.sort_values('step').reset_index(drop=True)

    def detect_divergence(self, losses: np.ndarray) -> Dict:
        """Detect loss divergence points."""
        divergences = []

        # Calculate moving average and std
        window = self.window_size
        moving_avg = pd.Series(losses).rolling(window=window, center=True).mean()
        moving_std = pd.Series(losses).rolling(window=window, center=True).std()

        # Detect sudden increases
        for i in range(window, len(losses) - window):
            current = losses[i]
            avg_before = moving_avg[i - window]
            std_before = moving_std[i - window]

            if std_before > 0:
                z_score = (current - avg_before) / std_before
                if z_score > 3.0 and current > avg_before * (1 + self.divergence_threshold):
                    divergences.append({
                        'step': i,
                        'loss': current,
                        'avg_before': avg_before,
                        'increase_pct': ((current - avg_before) / avg_before) * 100,
                        'z_score': z_score
                    })

        return {
            'count': len(divergences),
            'points': divergences,
            'severity': 'critical' if divergences else 'none'
        }

    def detect_plateau(self, losses: np.ndarray, min_length: int = 10) -> Dict:
        """Detect loss plateau regions."""
        plateaus = []

        # Calculate derivatives
        derivatives = np.gradient(losses)
        abs_derivatives = np.abs(derivatives)

        # Threshold for "flat" region
        threshold = np.percentile(abs_derivatives, 10)

        # Find consecutive flat regions
        is_flat = abs_derivatives < threshold
        plateau_start = None

        for i, flat in enumerate(is_flat):
            if flat and plateau_start is None:
                plateau_start = i
            elif not flat and plateau_start is not None:
                length = i - plateau_start
                if length >= min_length:
                    plateaus.append({
                        'start': plateau_start,
                        'end': i,
                        'length': length,
                        'avg_loss': np.mean(losses[plateau_start:i])
                    })
                plateau_start = None

        return {
            'count': len(plateaus),
            'regions': plateaus,
            'total_flat_steps': sum(p['length'] for p in plateaus)
        }

    def analyze_noise(self, losses: np.ndarray) -> Dict:
        """Analyze noise characteristics in loss curve."""
        # Smooth the curve
        if len(losses) > 11:
            smoothed = savgol_filter(losses, window_length=11, polyorder=3)
        else:
            smoothed = losses

        # Calculate residuals (noise)
        noise = losses - smoothed

        return {
            'std': float(np.std(noise)),
            'mean': float(np.mean(noise)),
            'snr': float(np.mean(smoothed) / (np.std(noise) + 1e-8)),
            'autocorrelation': float(np.corrcoef(noise[:-1], noise[1:])[0, 1])
        }

    def detect_anomalies(self, losses: np.ndarray, threshold: float = 3.0) -> Dict:
        """Detect anomalous loss values using z-score."""
        mean = np.mean(losses)
        std = np.std(losses)

        z_scores = np.abs((losses - mean) / (std + 1e-8))
        anomalies = np.where(z_scores > threshold)[0]

        return {
            'count': len(anomalies),
            'indices': anomalies.tolist(),
            'values': losses[anomalies].tolist(),
            'z_scores': z_scores[anomalies].tolist()
        }

    def analyze(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive loss curve analysis."""
        losses = df['loss'].values

        analysis = {
            'summary': {
                'total_steps': len(losses),
                'min_loss': float(np.min(losses)),
                'max_loss': float(np.max(losses)),
                'final_loss': float(losses[-1]),
                'mean_loss': float(np.mean(losses)),
                'std_loss': float(np.std(losses))
            },
            'trend': self._analyze_trend(losses),
            'divergence': self.detect_divergence(losses),
            'plateau': self.detect_plateau(losses),
            'noise': self.analyze_noise(losses),
            'anomalies': self.detect_anomalies(losses),
            'recommendations': []
        }

        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def _analyze_trend(self, losses: np.ndarray) -> Dict:
        """Analyze overall trend of loss curve."""
        x = np.arange(len(losses))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, losses)

        return {
            'slope': float(slope),
            'direction': 'decreasing' if slope < 0 else 'increasing',
            'r_squared': float(r_value ** 2),
            'significance': 'significant' if p_value < 0.05 else 'not significant'
        }

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate debugging recommendations based on analysis."""
        recommendations = []

        # Check divergence
        if analysis['divergence']['count'] > 0:
            div = analysis['divergence']['points'][0]
            recommendations.append(
                f"⚠️ CRITICAL: Loss diverged at step {div['step']} "
                f"({div['increase_pct']:.1f}% increase). "
                f"Consider reducing learning rate or checking for data issues."
            )

        # Check plateau
        if analysis['plateau']['count'] > 0:
            plateau = analysis['plateau']['regions'][0]
            recommendations.append(
                f"⏸️ Loss plateau detected from step {plateau['start']} to {plateau['end']}. "
                f"Consider increasing learning rate or model capacity."
            )

        # Check trend
        if analysis['trend']['direction'] == 'increasing':
            recommendations.append(
                "📈 Overall increasing trend - model not learning. "
                "Check data preprocessing, learning rate, and model initialization."
            )

        # Check noise
        if analysis['noise']['snr'] < 5:
            recommendations.append(
                f"📊 High noise (SNR={analysis['noise']['snr']:.2f}). "
                f"Consider larger batch size or gradient accumulation."
            )

        # Check anomalies
        if analysis['anomalies']['count'] > len(analysis['summary']['total_steps']) * 0.05:
            recommendations.append(
                f"⚡ {analysis['anomalies']['count']} anomalous loss spikes detected. "
                f"Check for data corruption or gradient clipping."
            )

        if not recommendations:
            recommendations.append("✅ Loss curve looks healthy. Continue monitoring.")

        return recommendations

    def visualize(self, df: pd.DataFrame, output_path: Path, analysis: Dict):
        """Create comprehensive loss visualization."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        losses = df['loss'].values
        steps = df['step'].values

        # Plot 1: Raw loss curve with anomalies
        ax1 = axes[0]
        ax1.plot(steps, losses, 'b-', alpha=0.6, label='Loss')

        # Highlight divergences
        if analysis['divergence']['count'] > 0:
            for div in analysis['divergence']['points']:
                ax1.axvline(steps[div['step']], color='red', linestyle='--', alpha=0.5)
                ax1.scatter([steps[div['step']]], [div['loss']], color='red', s=100,
                           marker='x', label='Divergence')

        # Highlight plateaus
        if analysis['plateau']['count'] > 0:
            for plateau in analysis['plateau']['regions']:
                ax1.axvspan(steps[plateau['start']], steps[plateau['end']],
                           alpha=0.2, color='yellow')

        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Smoothed loss with confidence interval
        ax2 = axes[1]
        if len(losses) > 11:
            smoothed = savgol_filter(losses, window_length=11, polyorder=3)
        else:
            smoothed = losses

        ax2.plot(steps, losses, 'b.', alpha=0.3, markersize=2, label='Raw')
        ax2.plot(steps, smoothed, 'r-', linewidth=2, label='Smoothed')
        ax2.fill_between(steps,
                         smoothed - analysis['noise']['std'],
                         smoothed + analysis['noise']['std'],
                         alpha=0.2, color='gray', label='±1 std')

        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Smoothed Loss with Noise Band')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Loss gradient (rate of change)
        ax3 = axes[2]
        gradient = np.gradient(losses)
        ax3.plot(steps, gradient, 'g-', alpha=0.7)
        ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Loss Gradient')
        ax3.set_title('Rate of Loss Change')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze ML training loss curves')
    parser.add_argument('--log-file', type=Path, help='Training log file')
    parser.add_argument('--csv', type=Path, help='CSV file with loss metrics')
    parser.add_argument('--output', type=Path, default='loss_analysis.png',
                       help='Output visualization path')
    parser.add_argument('--window', type=int, default=5,
                       help='Window size for moving average')
    parser.add_argument('--detect-divergence', action='store_true',
                       help='Focus on divergence detection')
    parser.add_argument('--json-output', type=Path,
                       help='Save analysis results as JSON')

    args = parser.parse_args()

    # Load data
    analyzer = LossAnalyzer(window_size=args.window)

    if args.log_file:
        df = analyzer.parse_log_file(args.log_file)
    elif args.csv:
        df = pd.read_csv(args.csv)
    else:
        print("Error: Must provide --log-file or --csv")
        sys.exit(1)

    # Analyze
    print("🔍 Analyzing loss curve...")
    analysis = analyzer.analyze(df)

    # Print summary
    print("\n📊 Analysis Summary:")
    print(f"Total steps: {analysis['summary']['total_steps']}")
    print(f"Min loss: {analysis['summary']['min_loss']:.6f}")
    print(f"Final loss: {analysis['summary']['final_loss']:.6f}")
    print(f"Trend: {analysis['trend']['direction']} (slope={analysis['trend']['slope']:.6e})")

    print("\n💡 Recommendations:")
    for rec in analysis['recommendations']:
        print(f"  {rec}")

    # Save JSON if requested
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n💾 Analysis saved to {args.json_output}")

    # Create visualization
    analyzer.visualize(df, args.output, analysis)


if __name__ == '__main__':
    main()
