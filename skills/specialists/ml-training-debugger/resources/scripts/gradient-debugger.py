#!/usr/bin/env python3
"""
Gradient Flow Debugger

Analyzes gradient flow through neural network layers to identify vanishing/exploding
gradients and provide diagnostic insights for training issues.

Usage:
    python gradient-debugger.py --checkpoint model.pt --config config.yaml
    python gradient-debugger.py --checkpoint model.pt --layer-wise --output grad_flow.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import defaultdict


class GradientDebugger:
    """Analyzes gradient flow through neural network layers."""

    def __init__(self, model: nn.Module, vanishing_threshold: float = 1e-6,
                 exploding_threshold: float = 100.0):
        self.model = model
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.gradient_stats = defaultdict(list)

    def register_hooks(self):
        """Register backward hooks to capture gradients."""
        def grad_hook(name):
            def hook(grad):
                if grad is not None:
                    self.gradient_stats[name].append({
                        'mean': grad.abs().mean().item(),
                        'std': grad.std().item(),
                        'max': grad.abs().max().item(),
                        'min': grad.abs().min().item(),
                        'norm': grad.norm().item(),
                        'num_zeros': (grad == 0).sum().item(),
                        'num_nans': torch.isnan(grad).sum().item(),
                        'num_infs': torch.isinf(grad).sum().item()
                    })
                return grad
            return hook

        # Register hooks for all parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(grad_hook(name))

    def analyze_layer_gradients(self, checkpoint_path: Path) -> Dict:
        """Analyze gradients from a model checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        analysis = {
            'layers': {},
            'summary': {
                'total_layers': 0,
                'vanishing_layers': 0,
                'exploding_layers': 0,
                'healthy_layers': 0
            },
            'issues': [],
            'recommendations': []
        }

        # Analyze each layer
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad

                stats = {
                    'mean_abs': grad.abs().mean().item(),
                    'std': grad.std().item(),
                    'max_abs': grad.abs().max().item(),
                    'min_abs': grad.abs().min().item(),
                    'norm': grad.norm().item(),
                    'sparsity': (grad == 0).float().mean().item(),
                    'has_nans': torch.isnan(grad).any().item(),
                    'has_infs': torch.isinf(grad).any().item()
                }

                # Classify gradient health
                if stats['has_nans'] or stats['has_infs']:
                    status = 'corrupt'
                    analysis['issues'].append(
                        f"⚠️ Layer '{name}': Gradient contains NaN/Inf values"
                    )
                elif stats['mean_abs'] < self.vanishing_threshold:
                    status = 'vanishing'
                    analysis['summary']['vanishing_layers'] += 1
                    analysis['issues'].append(
                        f"🔻 Layer '{name}': Vanishing gradient (mean={stats['mean_abs']:.2e})"
                    )
                elif stats['max_abs'] > self.exploding_threshold:
                    status = 'exploding'
                    analysis['summary']['exploding_layers'] += 1
                    analysis['issues'].append(
                        f"🔺 Layer '{name}': Exploding gradient (max={stats['max_abs']:.2e})"
                    )
                else:
                    status = 'healthy'
                    analysis['summary']['healthy_layers'] += 1

                stats['status'] = status
                analysis['layers'][name] = stats
                analysis['summary']['total_layers'] += 1

        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def analyze_gradient_flow(self, dummy_input: torch.Tensor,
                              dummy_target: torch.Tensor,
                              loss_fn: nn.Module) -> Dict:
        """Analyze gradient flow through a forward/backward pass."""
        self.gradient_stats.clear()
        self.register_hooks()

        # Forward pass
        self.model.train()
        output = self.model(dummy_input)
        loss = loss_fn(output, dummy_target)

        # Backward pass
        loss.backward()

        # Aggregate statistics
        analysis = {
            'layers': {},
            'flow_pattern': self._analyze_flow_pattern(),
            'bottlenecks': [],
            'recommendations': []
        }

        for name, stats_list in self.gradient_stats.items():
            if stats_list:
                stats = stats_list[-1]  # Use latest
                analysis['layers'][name] = stats

                # Detect bottlenecks
                if stats['mean'] < self.vanishing_threshold:
                    analysis['bottlenecks'].append({
                        'layer': name,
                        'type': 'vanishing',
                        'severity': 'high' if stats['mean'] < 1e-10 else 'medium'
                    })
                elif stats['max'] > self.exploding_threshold:
                    analysis['bottlenecks'].append({
                        'layer': name,
                        'type': 'exploding',
                        'severity': 'high' if stats['max'] > 1000 else 'medium'
                    })

        analysis['recommendations'] = self._generate_flow_recommendations(analysis)

        return analysis

    def _analyze_flow_pattern(self) -> str:
        """Determine overall gradient flow pattern."""
        if not self.gradient_stats:
            return 'unknown'

        # Calculate gradient magnitude trend from output to input
        layer_names = list(self.gradient_stats.keys())
        if len(layer_names) < 2:
            return 'insufficient_data'

        # Get mean gradients in reverse order (output -> input)
        means = [self.gradient_stats[name][-1]['mean']
                for name in reversed(layer_names)]

        # Analyze trend
        if all(m < self.vanishing_threshold for m in means[-3:]):
            return 'severe_vanishing'
        elif all(m > m_prev for m, m_prev in zip(means[1:], means[:-1])):
            return 'monotonic_increase'
        elif all(m < m_prev for m, m_prev in zip(means[1:], means[:-1])):
            return 'monotonic_decrease'
        else:
            return 'mixed'

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on gradient analysis."""
        recommendations = []

        vanishing_pct = (analysis['summary']['vanishing_layers'] /
                        max(analysis['summary']['total_layers'], 1)) * 100
        exploding_pct = (analysis['summary']['exploding_layers'] /
                        max(analysis['summary']['total_layers'], 1)) * 100

        if vanishing_pct > 30:
            recommendations.append(
                f"⚠️ {vanishing_pct:.1f}% layers have vanishing gradients. "
                f"Consider: (1) Different activation functions (ReLU→GELU), "
                f"(2) Residual connections, (3) Layer normalization, "
                f"(4) Xavier/He initialization"
            )

        if exploding_pct > 20:
            recommendations.append(
                f"⚠️ {exploding_pct:.1f}% layers have exploding gradients. "
                f"Consider: (1) Gradient clipping (max_norm=1.0), "
                f"(2) Lower learning rate, (3) Batch normalization, "
                f"(4) Weight initialization review"
            )

        # Check for specific patterns
        for name, stats in analysis['layers'].items():
            if stats.get('has_nans') or stats.get('has_infs'):
                recommendations.append(
                    f"🚨 CRITICAL: Layer '{name}' has NaN/Inf gradients. "
                    f"Check for: (1) Division by zero, (2) Overflow in loss, "
                    f"(3) Incompatible dtypes"
                )

            if stats.get('sparsity', 0) > 0.9:
                recommendations.append(
                    f"📊 Layer '{name}' has {stats['sparsity']*100:.1f}% zero gradients. "
                    f"Consider: (1) Dead ReLUs → Leaky ReLU, (2) Check data distribution"
                )

        if not recommendations:
            recommendations.append("✅ Gradient flow appears healthy across all layers")

        return recommendations

    def _generate_flow_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on flow analysis."""
        recommendations = []

        pattern = analysis['flow_pattern']

        if pattern == 'severe_vanishing':
            recommendations.append(
                "🔻 Severe gradient vanishing detected. "
                "Immediate actions: (1) Add skip connections, "
                "(2) Use GELU/Swish activations, (3) Apply layer normalization"
            )
        elif pattern == 'monotonic_decrease':
            recommendations.append(
                "📉 Gradients decreasing monotonically towards input layers. "
                "Consider: (1) Residual connections, (2) Gradient checkpointing, "
                "(3) Learning rate warmup"
            )

        if analysis['bottlenecks']:
            critical = [b for b in analysis['bottlenecks'] if b['severity'] == 'high']
            if critical:
                recommendations.append(
                    f"🎯 {len(critical)} critical gradient bottlenecks identified. "
                    f"Focus debugging on: {', '.join(b['layer'] for b in critical[:3])}"
                )

        return recommendations

    def visualize_gradient_flow(self, analysis: Dict, output_path: Path):
        """Create visualization of gradient flow."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        layer_names = list(analysis['layers'].keys())

        # Plot 1: Gradient magnitudes per layer
        ax1 = axes[0]
        means = [analysis['layers'][name]['mean'] for name in layer_names]
        maxs = [analysis['layers'][name]['max'] for name in layer_names]

        x = np.arange(len(layer_names))
        ax1.bar(x - 0.2, means, 0.4, label='Mean |gradient|', alpha=0.7)
        ax1.bar(x + 0.2, maxs, 0.4, label='Max |gradient|', alpha=0.7)

        ax1.axhline(self.vanishing_threshold, color='red', linestyle='--',
                   alpha=0.5, label='Vanishing threshold')
        ax1.axhline(self.exploding_threshold, color='orange', linestyle='--',
                   alpha=0.5, label='Exploding threshold')

        ax1.set_yscale('log')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layer_names, rotation=45, ha='right')
        ax1.set_ylabel('Gradient Magnitude (log scale)')
        ax1.set_title('Gradient Magnitudes per Layer')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Gradient norms
        ax2 = axes[1]
        norms = [analysis['layers'][name]['norm'] for name in layer_names]

        ax2.plot(x, norms, 'o-', linewidth=2, markersize=8)
        ax2.fill_between(x, 0, norms, alpha=0.3)

        # Highlight bottlenecks
        if 'bottlenecks' in analysis:
            for bottleneck in analysis['bottlenecks']:
                idx = layer_names.index(bottleneck['layer'])
                color = 'red' if bottleneck['severity'] == 'high' else 'orange'
                ax2.scatter([idx], [norms[idx]], s=200, color=color,
                           marker='x', linewidths=3, zorder=5)

        ax2.set_xticks(x)
        ax2.set_xticklabels(layer_names, rotation=45, ha='right')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title('Gradient Flow Pattern (Norm per Layer)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Gradient flow visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Debug gradient flow in neural networks')
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Model checkpoint file')
    parser.add_argument('--config', type=Path, help='Model config file')
    parser.add_argument('--layer-wise', action='store_true',
                       help='Perform layer-wise analysis')
    parser.add_argument('--output', type=Path, default='grad_flow.png',
                       help='Output visualization path')
    parser.add_argument('--vanishing-threshold', type=float, default=1e-6,
                       help='Threshold for vanishing gradients')
    parser.add_argument('--exploding-threshold', type=float, default=100.0,
                       help='Threshold for exploding gradients')
    parser.add_argument('--json-output', type=Path,
                       help='Save analysis as JSON')

    args = parser.parse_args()

    # Load model (placeholder - user would provide their model)
    print("⚠️ Note: This script requires model definition. Loading checkpoint only.")

    # For demonstration, create a simple model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    debugger = GradientDebugger(
        model,
        vanishing_threshold=args.vanishing_threshold,
        exploding_threshold=args.exploding_threshold
    )

    print("🔍 Analyzing gradient flow...")
    analysis = debugger.analyze_layer_gradients(args.checkpoint)

    # Print summary
    print("\n📊 Gradient Flow Analysis:")
    print(f"Total layers: {analysis['summary']['total_layers']}")
    print(f"Healthy layers: {analysis['summary']['healthy_layers']}")
    print(f"Vanishing gradients: {analysis['summary']['vanishing_layers']}")
    print(f"Exploding gradients: {analysis['summary']['exploding_layers']}")

    if analysis['issues']:
        print("\n⚠️ Issues Detected:")
        for issue in analysis['issues']:
            print(f"  {issue}")

    print("\n💡 Recommendations:")
    for rec in analysis['recommendations']:
        print(f"  {rec}")

    # Save JSON if requested
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n💾 Analysis saved to {args.json_output}")

    # Create visualization
    debugger.visualize_gradient_flow(analysis, args.output)


if __name__ == '__main__':
    main()
