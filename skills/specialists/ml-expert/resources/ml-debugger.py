#!/usr/bin/env python3
"""
ML-Specific Debugging and Diagnostics Tool

Specialized debugging utilities for machine learning workflows including
training diagnosis, gradient analysis, activation monitoring, and
performance profiling.

Usage:
    python ml-debugger.py --diagnose training_log.txt --output diagnosis.json
    python ml-debugger.py --profile model.py --batch-size 32
    python ml-debugger.py --analyze-gradients checkpoints/ --layer transformer.attention
"""

import argparse
import json
import re
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np


class MLDebugger:
    """ML-specific debugging and diagnostic tool."""

    def __init__(self):
        self.diagnostics = {}
        self.issues = []
        self.recommendations = []

    def diagnose_training_log(self, log_path: Path) -> Dict:
        """Analyze training logs for common issues."""
        with open(log_path) as f:
            log_content = f.read()

        diagnosis = {
            'loss_pattern': self.analyze_loss_pattern(log_content),
            'gradient_issues': self.detect_gradient_issues(log_content),
            'nan_inf_detection': self.detect_nan_inf(log_content),
            'convergence_status': self.analyze_convergence(log_content),
            'resource_usage': self.analyze_resource_usage(log_content)
        }

        # Generate recommendations
        self.generate_recommendations(diagnosis)
        diagnosis['recommendations'] = self.recommendations

        return diagnosis

    def analyze_loss_pattern(self, log_content: str) -> Dict:
        """Analyze loss values for patterns."""
        # Extract loss values
        loss_pattern = re.compile(r'loss[:\s=]+([0-9.]+)')
        losses = [float(match) for match in loss_pattern.findall(log_content)]

        if not losses:
            return {'status': 'no_losses_found'}

        analysis = {
            'min_loss': min(losses),
            'max_loss': max(losses),
            'final_loss': losses[-1],
            'total_iterations': len(losses),
            'trend': 'unknown'
        }

        # Analyze trend
        if len(losses) >= 10:
            early_avg = np.mean(losses[:len(losses)//3])
            late_avg = np.mean(losses[-len(losses)//3:])

            if late_avg < early_avg * 0.5:
                analysis['trend'] = 'improving'
            elif late_avg > early_avg * 1.5:
                analysis['trend'] = 'degrading'
            elif abs(late_avg - early_avg) / early_avg < 0.1:
                analysis['trend'] = 'plateau'
            else:
                analysis['trend'] = 'unstable'

        # Check for explosion
        if any(loss > 1e6 for loss in losses):
            analysis['explosion_detected'] = True
            self.issues.append('Loss explosion detected')

        # Check for stagnation
        if len(losses) >= 50:
            recent_variance = np.var(losses[-50:])
            if recent_variance < 1e-6:
                analysis['stagnation_detected'] = True
                self.issues.append('Loss stagnation detected')

        return analysis

    def detect_gradient_issues(self, log_content: str) -> Dict:
        """Detect gradient-related issues."""
        issues = {
            'vanishing_gradients': False,
            'exploding_gradients': False,
            'zero_gradients': False
        }

        # Check for vanishing gradient warnings
        if 'gradient' in log_content.lower() and 'small' in log_content.lower():
            issues['vanishing_gradients'] = True
            self.issues.append('Possible vanishing gradients')

        # Check for exploding gradient warnings
        if 'gradient' in log_content.lower() and 'large' in log_content.lower():
            issues['exploding_gradients'] = True
            self.issues.append('Possible exploding gradients')

        # Extract gradient norms if available
        grad_norm_pattern = re.compile(r'grad_norm[:\s=]+([0-9.e+-]+)')
        grad_norms = [float(match) for match in grad_norm_pattern.findall(log_content)]

        if grad_norms:
            issues['max_grad_norm'] = max(grad_norms)
            issues['min_grad_norm'] = min(grad_norms)

            if max(grad_norms) > 100:
                issues['exploding_gradients'] = True
            if min(grad_norms) < 1e-7:
                issues['vanishing_gradients'] = True

        return issues

    def detect_nan_inf(self, log_content: str) -> Dict:
        """Detect NaN or Inf values."""
        detection = {
            'nan_detected': 'nan' in log_content.lower(),
            'inf_detected': 'inf' in log_content.lower(),
            'occurrences': []
        }

        if detection['nan_detected']:
            self.issues.append('NaN values detected in training')
        if detection['inf_detected']:
            self.issues.append('Inf values detected in training')

        return detection

    def analyze_convergence(self, log_content: str) -> Dict:
        """Analyze training convergence."""
        # Extract accuracy/metrics if available
        acc_pattern = re.compile(r'acc(?:uracy)?[:\s=]+([0-9.]+)')
        accuracies = [float(match) for match in acc_pattern.findall(log_content)]

        convergence = {
            'converged': False,
            'plateau_reached': False,
            'oscillating': False
        }

        if accuracies and len(accuracies) >= 10:
            recent_acc = accuracies[-10:]
            variance = np.var(recent_acc)

            if variance < 0.01 and recent_acc[-1] > 0.5:
                convergence['converged'] = True
                convergence['plateau_reached'] = True
            elif variance > 0.1:
                convergence['oscillating'] = True
                self.issues.append('Training metrics oscillating')

        return convergence

    def analyze_resource_usage(self, log_content: str) -> Dict:
        """Analyze resource usage patterns."""
        usage = {
            'oom_detected': False,
            'slow_training': False
        }

        # Check for OOM errors
        if 'out of memory' in log_content.lower() or 'oom' in log_content.lower():
            usage['oom_detected'] = True
            self.issues.append('Out of memory errors detected')

        # Check for slow training warnings
        if 'slow' in log_content.lower() or 'timeout' in log_content.lower():
            usage['slow_training'] = True
            self.issues.append('Slow training detected')

        return usage

    def generate_recommendations(self, diagnosis: Dict):
        """Generate recommendations based on diagnosis."""
        loss_analysis = diagnosis['loss_pattern']
        grad_issues = diagnosis['gradient_issues']

        # Loss-based recommendations
        if loss_analysis.get('explosion_detected'):
            self.recommendations.append({
                'issue': 'Loss explosion',
                'fix': 'Reduce learning rate by 10x or enable gradient clipping',
                'priority': 'high'
            })

        if loss_analysis.get('stagnation_detected'):
            self.recommendations.append({
                'issue': 'Loss stagnation',
                'fix': 'Increase learning rate, add learning rate scheduling, or check data quality',
                'priority': 'medium'
            })

        # Gradient-based recommendations
        if grad_issues.get('vanishing_gradients'):
            self.recommendations.append({
                'issue': 'Vanishing gradients',
                'fix': 'Use residual connections, layer normalization, or reduce model depth',
                'priority': 'high'
            })

        if grad_issues.get('exploding_gradients'):
            self.recommendations.append({
                'issue': 'Exploding gradients',
                'fix': 'Enable gradient clipping (max_norm=1.0) or reduce learning rate',
                'priority': 'high'
            })

        # NaN/Inf recommendations
        if diagnosis['nan_inf_detection']['nan_detected']:
            self.recommendations.append({
                'issue': 'NaN values',
                'fix': 'Check for division by zero, log of negative numbers, or unstable operations',
                'priority': 'critical'
            })

    def profile_model(self, model_path: Path, batch_size: int = 32) -> Dict:
        """Profile model performance."""
        # This would load and profile an actual model
        # Simplified for demonstration
        profile = {
            'parameter_count': 0,
            'memory_usage_mb': 0,
            'inference_time_ms': 0,
            'bottlenecks': []
        }

        return profile

    def analyze_gradients(self, checkpoint_dir: Path, layer_name: Optional[str] = None) -> Dict:
        """Analyze gradient flow through network."""
        gradient_analysis = {
            'layers_analyzed': [],
            'gradient_magnitudes': {},
            'gradient_flow_issues': []
        }

        # Load checkpoints and analyze
        # Simplified for demonstration

        return gradient_analysis

    def export_report(self, output_path: Path):
        """Export comprehensive debugging report."""
        report = {
            'timestamp': str(Path.ctime(output_path)),
            'issues_found': self.issues,
            'recommendations': self.recommendations,
            'diagnostics': self.diagnostics
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='ML Debugging Tool')
    parser.add_argument('--diagnose', type=Path, help='Training log to diagnose')
    parser.add_argument('--output', type=Path, help='Output diagnosis file')
    parser.add_argument('--profile', type=Path, help='Model file to profile')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for profiling')
    parser.add_argument('--analyze-gradients', type=Path, help='Checkpoint directory')
    parser.add_argument('--layer', type=str, help='Specific layer to analyze')

    args = parser.parse_args()

    debugger = MLDebugger()

    if args.diagnose:
        print(f"Diagnosing training log: {args.diagnose}")
        diagnosis = debugger.diagnose_training_log(args.diagnose)

        print("\n=== Diagnosis Summary ===")
        print(f"Issues found: {len(debugger.issues)}")
        for issue in debugger.issues:
            print(f"  - {issue}")

        print(f"\nRecommendations: {len(debugger.recommendations)}")
        for rec in debugger.recommendations:
            print(f"  [{rec['priority'].upper()}] {rec['issue']}: {rec['fix']}")

        if args.output:
            debugger.diagnostics = diagnosis
            debugger.export_report(args.output)
            print(f"\nFull report saved to: {args.output}")

    elif args.profile:
        print(f"Profiling model: {args.profile}")
        profile = debugger.profile_model(args.profile, args.batch_size)
        print(json.dumps(profile, indent=2))

    elif args.analyze_gradients:
        print(f"Analyzing gradients: {args.analyze_gradients}")
        analysis = debugger.analyze_gradients(args.analyze_gradients, args.layer)
        print(json.dumps(analysis, indent=2))


if __name__ == '__main__':
    main()
