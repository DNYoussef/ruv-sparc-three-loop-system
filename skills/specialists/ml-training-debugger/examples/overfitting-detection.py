#!/usr/bin/env python3
"""
Example: Overfitting Detection and Mitigation

This example demonstrates detecting overfitting through train/val gap analysis
and applying regularization techniques to improve generalization.

Scenario:
    A neural network achieves perfect training accuracy but poor validation
    performance, indicating severe overfitting.

Workflow:
    1. Train baseline model with minimal regularization
    2. Detect overfitting via train/val gap analysis
    3. Apply regularization (dropout, L2, early stopping)
    4. Compare results and verify improved generalization

Usage:
    python overfitting-detection.py --mode baseline
    python overfitting-detection.py --mode regularized
    python overfitting-detection.py --mode all
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np


class OverfittingProneNetwork(nn.Module):
    """Large network prone to overfitting on small dataset."""

    def __init__(self, input_dim=50, output_dim=5):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class RegularizedNetwork(nn.Module):
    """Network with dropout and smaller capacity."""

    def __init__(self, input_dim=50, output_dim=5, dropout=0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def create_limited_dataset(num_samples=500, input_dim=50, output_dim=5):
    """Create small dataset to induce overfitting."""
    # Generate data with clear patterns
    X = torch.randn(num_samples, input_dim)

    # Add some structure to make learning possible
    y = (X[:, :5].sum(dim=1) > 0).long() + \
        (X[:, 5:10].sum(dim=1) > 0).long() * 2 + \
        (X[:, 10:15].sum(dim=1) > 0).long()

    y = y % output_dim  # Ensure valid class indices

    return TensorDataset(X, y)


def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion,
                       num_epochs=50, early_stopping_patience=None):
    """Train model and track train/val metrics."""
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                loss = criterion(output, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Print progress
        gap = abs(avg_train_loss - avg_val_loss) / avg_val_loss
        print(f"Epoch {epoch+1:2d}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train Acc: {train_accuracy:.3f}, Val Acc: {val_accuracy:.3f}, "
              f"Gap: {gap*100:.1f}%")

        # Early stopping
        if early_stopping_patience is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
                break

    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    }


def analyze_overfitting(metrics):
    """Analyze train/val gap to detect overfitting."""
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']

    # Calculate gaps
    gaps = []
    for t, v in zip(train_loss, val_loss):
        gap = abs(t - v) / (v + 1e-8)
        gaps.append(gap * 100)  # Convert to percentage

    # Detect overfitting onset
    onset_epoch = None
    gap_threshold = 15.0  # 15% gap

    for i, gap in enumerate(gaps):
        if gap > gap_threshold:
            onset_epoch = i
            break

    # Calculate severity
    max_gap = max(gaps)
    avg_gap = sum(gaps) / len(gaps)
    final_gap = gaps[-1]

    if max_gap > 50:
        severity = 'severe'
    elif max_gap > 30:
        severity = 'high'
    elif max_gap > 15:
        severity = 'moderate'
    else:
        severity = 'mild'

    analysis = {
        'detected': onset_epoch is not None,
        'onset_epoch': onset_epoch,
        'severity': severity,
        'max_gap': max_gap,
        'avg_gap': avg_gap,
        'final_gap': final_gap,
        'gaps': gaps
    }

    return analysis


def print_overfitting_report(analysis):
    """Print detailed overfitting analysis."""
    print("\n" + "=" * 70)
    print("OVERFITTING ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nDetected: {'YES' if analysis['detected'] else 'NO'}")

    if analysis['detected']:
        print(f"Onset Epoch: {analysis['onset_epoch']}")
        print(f"Severity: {analysis['severity'].upper()}")
        print(f"Max Train/Val Gap: {analysis['max_gap']:.1f}%")
        print(f"Average Gap: {analysis['avg_gap']:.1f}%")
        print(f"Final Gap: {analysis['final_gap']:.1f}%")

        print("\n💡 Recommendations:")

        if analysis['severity'] == 'severe':
            print("   CRITICAL OVERFITTING - Immediate action required:")
            print("   1. Reduce model capacity (fewer layers/units)")
            print("   2. Increase dropout rate to 0.4-0.5")
            print("   3. Add L2 regularization (weight_decay=1e-3)")
            print("   4. Collect more training data if possible")
            print("   5. Use data augmentation")

        elif analysis['severity'] == 'high':
            print("   HIGH OVERFITTING - Strong regularization needed:")
            print("   1. Add/increase dropout to 0.3-0.4")
            print("   2. Apply L2 regularization (weight_decay=1e-4)")
            print("   3. Use early stopping (patience=10)")
            print("   4. Consider batch normalization")

        elif analysis['severity'] == 'moderate':
            print("   MODERATE OVERFITTING - Regularization recommended:")
            print("   1. Add dropout (0.2-0.3)")
            print("   2. Apply light L2 regularization (weight_decay=1e-5)")
            print("   3. Monitor with early stopping")
            print("   4. Review model capacity")
    else:
        print("\n✅ No significant overfitting detected")
        print("   Training appears healthy")


def visualize_comparison(baseline_metrics, regularized_metrics, output_path):
    """Visualize baseline vs regularized training."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs_baseline = range(1, len(baseline_metrics['train_loss']) + 1)
    epochs_reg = range(1, len(regularized_metrics['train_loss']) + 1)

    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs_baseline, baseline_metrics['train_loss'], 'r-',
             label='Baseline Train', linewidth=2)
    ax1.plot(epochs_baseline, baseline_metrics['val_loss'], 'r--',
             label='Baseline Val', linewidth=2)
    ax1.plot(epochs_reg, regularized_metrics['train_loss'], 'g-',
             label='Regularized Train', linewidth=2)
    ax1.plot(epochs_reg, regularized_metrics['val_loss'], 'g--',
             label='Regularized Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(epochs_baseline, baseline_metrics['train_accuracy'], 'r-',
             label='Baseline Train', linewidth=2)
    ax2.plot(epochs_baseline, baseline_metrics['val_accuracy'], 'r--',
             label='Baseline Val', linewidth=2)
    ax2.plot(epochs_reg, regularized_metrics['train_accuracy'], 'g-',
             label='Regularized Train', linewidth=2)
    ax2.plot(epochs_reg, regularized_metrics['val_accuracy'], 'g--',
             label='Regularized Val', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Train/Val gap
    ax3 = axes[1, 0]
    baseline_gap = [abs(t - v) for t, v in zip(baseline_metrics['train_loss'],
                                                 baseline_metrics['val_loss'])]
    reg_gap = [abs(t - v) for t, v in zip(regularized_metrics['train_loss'],
                                           regularized_metrics['val_loss'])]
    ax3.plot(epochs_baseline, baseline_gap, 'r-', label='Baseline', linewidth=2)
    ax3.plot(epochs_reg, reg_gap, 'g-', label='Regularized', linewidth=2)
    ax3.axhline(0.15, color='orange', linestyle='--', alpha=0.5,
               label='Overfitting threshold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Train/Val Loss Gap')
    ax3.set_title('Overfitting Gap Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Val accuracy improvement
    ax4 = axes[1, 1]
    # Match lengths
    min_len = min(len(baseline_metrics['val_accuracy']),
                  len(regularized_metrics['val_accuracy']))
    improvement = [(r - b) * 100 for b, r in zip(
        baseline_metrics['val_accuracy'][:min_len],
        regularized_metrics['val_accuracy'][:min_len]
    )]
    ax4.plot(range(1, min_len + 1), improvement, 'b-', linewidth=2)
    ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Val Accuracy Improvement (%)')
    ax4.set_title('Regularization Benefit')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison visualization saved to {output_path}")


def run_baseline_experiment():
    """Run baseline experiment prone to overfitting."""
    print("=" * 70)
    print("BASELINE EXPERIMENT: Large Model, Minimal Regularization")
    print("=" * 70)

    # Create small dataset
    dataset = create_limited_dataset(num_samples=500)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    # Create large model
    model = OverfittingProneNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # No weight decay
    criterion = nn.CrossEntropyLoss()

    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Dataset: {len(train_data)} train, {len(val_data)} val samples")
    print(f"Regularization: None")
    print()

    # Train
    metrics = train_and_evaluate(model, train_loader, val_loader,
                                 optimizer, criterion, num_epochs=50)

    # Analyze overfitting
    analysis = analyze_overfitting(metrics)
    print_overfitting_report(analysis)

    # Save results
    results = {
        'metrics': metrics,
        'analysis': analysis,
        'model_state': model.state_dict()
    }
    torch.save(results, "baseline_overfitting_results.pt")

    return metrics, analysis


def run_regularized_experiment():
    """Run experiment with regularization."""
    print("\n" + "=" * 70)
    print("REGULARIZED EXPERIMENT: Dropout + L2 + Early Stopping")
    print("=" * 70)

    # Create same dataset
    dataset = create_limited_dataset(num_samples=500)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    # Create regularized model
    model = RegularizedNetwork(dropout=0.3)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2
    criterion = nn.CrossEntropyLoss()

    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Dataset: {len(train_data)} train, {len(val_data)} val samples")
    print(f"Regularization:")
    print(f"  - Dropout: 0.3")
    print(f"  - L2 (weight_decay): 1e-4")
    print(f"  - Early stopping: patience=10")
    print()

    # Train with early stopping
    metrics = train_and_evaluate(model, train_loader, val_loader,
                                 optimizer, criterion, num_epochs=50,
                                 early_stopping_patience=10)

    # Analyze overfitting
    analysis = analyze_overfitting(metrics)
    print_overfitting_report(analysis)

    # Save results
    results = {
        'metrics': metrics,
        'analysis': analysis,
        'model_state': model.state_dict()
    }
    torch.save(results, "regularized_overfitting_results.pt")

    return metrics, analysis


def main():
    parser = argparse.ArgumentParser(description='Overfitting Detection Example')
    parser.add_argument('--mode', choices=['baseline', 'regularized', 'all'],
                       default='all', help='Experiment mode')
    args = parser.parse_args()

    if args.mode in ['baseline', 'all']:
        baseline_metrics, baseline_analysis = run_baseline_experiment()

    if args.mode in ['regularized', 'all']:
        regularized_metrics, regularized_analysis = run_regularized_experiment()

    if args.mode == 'all':
        # Compare results
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)

        print(f"\nBaseline:")
        print(f"  Final Val Accuracy: {baseline_metrics['val_accuracy'][-1]:.3f}")
        print(f"  Overfitting Severity: {baseline_analysis['severity']}")
        print(f"  Max Train/Val Gap: {baseline_analysis['max_gap']:.1f}%")

        print(f"\nRegularized:")
        print(f"  Final Val Accuracy: {regularized_metrics['val_accuracy'][-1]:.3f}")
        print(f"  Overfitting Severity: {regularized_analysis['severity']}")
        print(f"  Max Train/Val Gap: {regularized_analysis['max_gap']:.1f}%")

        val_improvement = (regularized_metrics['val_accuracy'][-1] -
                          baseline_metrics['val_accuracy'][-1]) * 100
        gap_reduction = baseline_analysis['max_gap'] - regularized_analysis['max_gap']

        print(f"\n✅ Improvements:")
        print(f"  Val Accuracy: +{val_improvement:.1f}%")
        print(f"  Gap Reduction: -{gap_reduction:.1f}%")

        # Visualize
        visualize_comparison(baseline_metrics, regularized_metrics,
                           "overfitting_comparison.png")

    print("\n" + "=" * 70)
    print("Example complete! Generated files:")
    print("  - baseline_overfitting_results.pt")
    print("  - regularized_overfitting_results.pt")
    print("  - overfitting_comparison.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
