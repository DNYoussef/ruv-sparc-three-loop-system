#!/usr/bin/env python3
"""
Example: Convergence Debugging

This example demonstrates diagnosing and fixing convergence issues where
training loss plateaus or fails to decrease effectively.

Scenario:
    A model's training loss plateaus early at a suboptimal value, suggesting
    learning rate issues, architecture problems, or optimization challenges.

Workflow:
    1. Train baseline model that plateaus
    2. Analyze loss curves to identify plateau characteristics
    3. Apply fixes (learning rate scheduling, architecture changes)
    4. Compare convergence improvements

Usage:
    python convergence-debugging.py --mode baseline
    python convergence-debugging.py --mode fixed
    python convergence-debugging.py --mode all
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np


class PlateauProneNetwork(nn.Module):
    """Network architecture that tends to plateau."""

    def __init__(self, input_dim=100, output_dim=10):
        super().__init__()

        # Bottleneck architecture can cause plateaus
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),  # Tanh can saturate
            nn.Linear(64, 16),  # Severe bottleneck
            nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class ImprovedNetwork(nn.Module):
    """Improved architecture for better convergence."""

    def __init__(self, input_dim=100, output_dim=10):
        super().__init__()

        # Better capacity and activations
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def create_challenging_dataset(num_samples=2000, input_dim=100, output_dim=10):
    """Create dataset that requires good optimization."""
    # Generate data with complex patterns
    X = torch.randn(num_samples, input_dim)

    # Complex target function
    y = torch.zeros(num_samples, dtype=torch.long)
    for i in range(num_samples):
        # Non-linear combination of features
        score = (X[i, :20].sum() * X[i, 20:40].prod()**0.1 +
                X[i, 40:60].std() * 10 + X[i, 60:80].mean()**2)
        y[i] = int(abs(score) * 10) % output_dim

    return TensorDataset(X, y)


def train_model(model, dataloader, optimizer, criterion, scheduler=None, num_epochs=100):
    """Train model and track detailed metrics."""
    losses = []
    learning_rates = []
    gradient_norms = []

    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        num_batches = 0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()

            # Calculate gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            optimizer.step()

            epoch_loss += loss.item()
            epoch_grad_norm += total_norm
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_grad_norm = epoch_grad_norm / num_batches

        losses.append(avg_loss)
        gradient_norms.append(avg_grad_norm)

        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} - Loss: {avg_loss:.4f}, "
                  f"Grad Norm: {avg_grad_norm:.4f}, LR: {current_lr:.6f}")

    return losses, learning_rates, gradient_norms


def detect_plateau(losses, window=10, threshold=0.01):
    """Detect plateau regions in loss curve."""
    plateaus = []

    if len(losses) < window * 2:
        return plateaus

    for i in range(window, len(losses) - window):
        # Calculate improvement rate
        before_avg = sum(losses[i-window:i]) / window
        after_avg = sum(losses[i:i+window]) / window

        improvement = (before_avg - after_avg) / (before_avg + 1e-8)

        if abs(improvement) < threshold:
            plateaus.append({
                'start': i - window,
                'end': i + window,
                'loss': after_avg,
                'improvement': improvement
            })

    # Merge overlapping plateaus
    merged = []
    for plateau in plateaus:
        if not merged or plateau['start'] > merged[-1]['end']:
            merged.append(plateau)
        else:
            merged[-1]['end'] = plateau['end']

    return merged


def analyze_convergence(losses, learning_rates, gradient_norms):
    """Analyze convergence characteristics."""
    analysis = {
        'final_loss': losses[-1],
        'min_loss': min(losses),
        'convergence_rate': None,
        'plateaus': [],
        'gradient_health': None,
        'lr_effectiveness': None
    }

    # Detect plateaus
    plateaus = detect_plateau(losses)
    analysis['plateaus'] = plateaus

    # Calculate convergence rate (first 20 epochs)
    if len(losses) >= 20:
        initial_loss = losses[0]
        loss_20 = losses[19]
        analysis['convergence_rate'] = (initial_loss - loss_20) / initial_loss

    # Analyze gradient health
    avg_grad_norm = sum(gradient_norms) / len(gradient_norms)
    recent_grad_norm = sum(gradient_norms[-10:]) / 10

    if recent_grad_norm < 1e-5:
        analysis['gradient_health'] = 'vanishing'
    elif recent_grad_norm > 100:
        analysis['gradient_health'] = 'exploding'
    elif recent_grad_norm < avg_grad_norm * 0.1:
        analysis['gradient_health'] = 'decaying'
    else:
        analysis['gradient_health'] = 'healthy'

    # Analyze LR effectiveness
    final_lr = learning_rates[-1]
    initial_lr = learning_rates[0]

    if final_lr < initial_lr * 0.01:
        analysis['lr_effectiveness'] = 'too_aggressive_decay'
    elif final_lr == initial_lr and len(plateaus) > 0:
        analysis['lr_effectiveness'] = 'needs_scheduling'
    else:
        analysis['lr_effectiveness'] = 'appropriate'

    return analysis


def print_convergence_report(analysis):
    """Print detailed convergence analysis."""
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nFinal Loss: {analysis['final_loss']:.4f}")
    print(f"Best Loss: {analysis['min_loss']:.4f}")

    if analysis['convergence_rate'] is not None:
        print(f"Convergence Rate (first 20 epochs): {analysis['convergence_rate']*100:.1f}%")

    # Plateau analysis
    if analysis['plateaus']:
        print(f"\n⏸️  PLATEAUS DETECTED: {len(analysis['plateaus'])} regions")
        for i, plateau in enumerate(analysis['plateaus'][:3]):  # Show first 3
            print(f"   Plateau {i+1}: Epochs {plateau['start']}-{plateau['end']}, "
                  f"Loss: {plateau['loss']:.4f}")

        print("\n💡 Plateau Recommendations:")
        print("   1. Reduce learning rate when plateau detected")
        print("   2. Use learning rate scheduling (cosine annealing)")
        print("   3. Check if model has sufficient capacity")
        print("   4. Consider different optimizer (Adam → AdamW)")
    else:
        print("\n✅ No significant plateaus detected")

    # Gradient health
    print(f"\nGradient Health: {analysis['gradient_health']}")

    if analysis['gradient_health'] == 'vanishing':
        print("   ⚠️ Gradients vanishing - model not learning effectively")
        print("   Recommendations:")
        print("     - Change activation functions (Tanh → ReLU)")
        print("     - Add batch normalization")
        print("     - Check weight initialization")
    elif analysis['gradient_health'] == 'decaying':
        print("   ⚠️ Gradients decaying rapidly")
        print("   Recommendations:")
        print("     - Increase learning rate")
        print("     - Review architecture depth")

    # Learning rate effectiveness
    print(f"\nLearning Rate Strategy: {analysis['lr_effectiveness']}")

    if analysis['lr_effectiveness'] == 'needs_scheduling':
        print("   ⚠️ Fixed LR with plateaus - needs adaptive scheduling")
        print("   Recommendations:")
        print("     - Use ReduceLROnPlateau")
        print("     - Implement cosine annealing")
        print("     - Try warmup + decay schedule")


def visualize_convergence(baseline_results, fixed_results, output_path):
    """Visualize convergence comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs_baseline = range(1, len(baseline_results['losses']) + 1)
    epochs_fixed = range(1, len(fixed_results['losses']) + 1)

    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs_baseline, baseline_results['losses'], 'r-',
             label='Baseline', linewidth=2)
    ax1.plot(epochs_fixed, fixed_results['losses'], 'g-',
             label='Fixed', linewidth=2)

    # Mark plateaus
    for plateau in baseline_results['analysis']['plateaus']:
        ax1.axvspan(plateau['start'], plateau['end'],
                   alpha=0.2, color='red')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Convergence Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Learning rates
    ax2 = axes[0, 1]
    ax2.plot(epochs_baseline, baseline_results['learning_rates'], 'r-',
             label='Baseline', linewidth=2)
    ax2.plot(epochs_fixed, fixed_results['learning_rates'], 'g-',
             label='Fixed', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedules')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Gradient norms
    ax3 = axes[1, 0]
    ax3.plot(epochs_baseline, baseline_results['gradient_norms'], 'r-',
             label='Baseline', linewidth=2)
    ax3.plot(epochs_fixed, fixed_results['gradient_norms'], 'g-',
             label='Fixed', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Gradient Flow Health')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Loss improvement rate
    ax4 = axes[1, 1]

    # Calculate smoothed loss improvement
    window = 5
    baseline_smooth = np.convolve(baseline_results['losses'],
                                  np.ones(window)/window, mode='valid')
    fixed_smooth = np.convolve(fixed_results['losses'],
                              np.ones(window)/window, mode='valid')

    min_len = min(len(baseline_smooth), len(fixed_smooth))
    improvement = [(b - f) / b * 100 for b, f in
                  zip(baseline_smooth[:min_len], fixed_smooth[:min_len])]

    ax4.plot(range(window, window + len(improvement)), improvement, 'b-',
            linewidth=2)
    ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Improvement (%)')
    ax4.set_title('Relative Improvement')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Convergence visualization saved to {output_path}")


def run_baseline_experiment():
    """Run baseline experiment with convergence issues."""
    print("=" * 70)
    print("BASELINE EXPERIMENT: Plateau-Prone Configuration")
    print("=" * 70)

    # Create dataset
    dataset = create_challenging_dataset(num_samples=2000)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Create problematic model
    model = PlateauProneNetwork()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # No scheduling
    criterion = nn.CrossEntropyLoss()

    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Architecture: Bottleneck with Tanh activations")
    print(f"Optimizer: SGD with fixed LR=0.01")
    print(f"Scheduler: None\n")

    # Train
    losses, learning_rates, gradient_norms = train_model(
        model, dataloader, optimizer, criterion, num_epochs=100
    )

    # Analyze
    analysis = analyze_convergence(losses, learning_rates, gradient_norms)
    print_convergence_report(analysis)

    results = {
        'losses': losses,
        'learning_rates': learning_rates,
        'gradient_norms': gradient_norms,
        'analysis': analysis
    }

    torch.save(results, "baseline_convergence_results.pt")

    return results


def run_fixed_experiment():
    """Run experiment with convergence fixes."""
    print("\n" + "=" * 70)
    print("FIXED EXPERIMENT: Improved Architecture + LR Scheduling")
    print("=" * 70)

    # Create dataset
    dataset = create_challenging_dataset(num_samples=2000)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Create improved model
    model = ImprovedNetwork()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Add learning rate scheduling
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss()

    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Architecture: Wider layers + ReLU + BatchNorm")
    print(f"Optimizer: AdamW with weight_decay=1e-4")
    print(f"Scheduler: Cosine Annealing (eta_min=1e-6)\n")

    # Train
    losses, learning_rates, gradient_norms = train_model(
        model, dataloader, optimizer, criterion, scheduler, num_epochs=100
    )

    # Analyze
    analysis = analyze_convergence(losses, learning_rates, gradient_norms)
    print_convergence_report(analysis)

    results = {
        'losses': losses,
        'learning_rates': learning_rates,
        'gradient_norms': gradient_norms,
        'analysis': analysis
    }

    torch.save(results, "fixed_convergence_results.pt")

    return results


def main():
    parser = argparse.ArgumentParser(description='Convergence Debugging Example')
    parser.add_argument('--mode', choices=['baseline', 'fixed', 'all'],
                       default='all', help='Experiment mode')
    args = parser.parse_args()

    if args.mode in ['baseline', 'all']:
        baseline_results = run_baseline_experiment()

    if args.mode in ['fixed', 'all']:
        fixed_results = run_fixed_experiment()

    if args.mode == 'all':
        # Compare
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)

        print(f"\nBaseline:")
        print(f"  Final Loss: {baseline_results['analysis']['final_loss']:.4f}")
        print(f"  Best Loss: {baseline_results['analysis']['min_loss']:.4f}")
        print(f"  Plateaus: {len(baseline_results['analysis']['plateaus'])}")

        print(f"\nFixed:")
        print(f"  Final Loss: {fixed_results['analysis']['final_loss']:.4f}")
        print(f"  Best Loss: {fixed_results['analysis']['min_loss']:.4f}")
        print(f"  Plateaus: {len(fixed_results['analysis']['plateaus'])}")

        improvement = (baseline_results['analysis']['final_loss'] -
                      fixed_results['analysis']['final_loss']) / \
                     baseline_results['analysis']['final_loss'] * 100

        print(f"\n✅ Final Loss Improvement: {improvement:.1f}%")

        # Visualize
        visualize_convergence(baseline_results, fixed_results,
                            "convergence_comparison.png")

    print("\n" + "=" * 70)
    print("Example complete! Generated files:")
    print("  - baseline_convergence_results.pt")
    print("  - fixed_convergence_results.pt")
    print("  - convergence_comparison.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
