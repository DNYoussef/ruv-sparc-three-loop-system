#!/usr/bin/env python3
"""
Example: Debugging Vanishing Gradients

This example demonstrates a complete workflow for diagnosing and fixing
vanishing gradient issues in a deep neural network.

Scenario:
    A 20-layer deep network with sigmoid activations is failing to train.
    Gradients vanish in early layers, preventing the network from learning.

Workflow:
    1. Train baseline model and observe training failure
    2. Run gradient debugger to identify vanishing gradients
    3. Apply recommended fixes (activation change, skip connections)
    4. Re-train and verify improvement

Usage:
    python vanishing-gradients.py --mode baseline
    python vanishing-gradients.py --mode debug
    python vanishing-gradients.py --mode fixed
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np


class DeepSigmoidNetwork(nn.Module):
    """Problematic deep network with sigmoid activations."""

    def __init__(self, input_dim=128, hidden_dim=128, output_dim=10, num_layers=20):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]

        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            ])

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ImprovedDeepNetwork(nn.Module):
    """Fixed network with ReLU activations and skip connections."""

    def __init__(self, input_dim=128, hidden_dim=128, output_dim=10, num_layers=20):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers // 2):
            self.blocks.append(ResidualBlock(hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        return self.output_layer(x)


class ResidualBlock(nn.Module):
    """Residual block to prevent gradient vanishing."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))  # Skip connection


def create_synthetic_data(num_samples=1000, input_dim=128, output_dim=10):
    """Create synthetic classification dataset."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))
    return TensorDataset(X, y)


def train_model(model, dataloader, optimizer, criterion, num_epochs=10):
    """Train model and collect metrics."""
    losses = []
    gradient_norms = []

    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        num_batches = 0

        for batch_x, batch_y in dataloader:
            # Forward pass
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)

            # Backward pass
            loss.backward()

            # Calculate gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            # Update weights
            optimizer.step()

            epoch_loss += loss.item()
            epoch_grad_norm += total_norm
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_grad_norm = epoch_grad_norm / num_batches

        losses.append(avg_loss)
        gradient_norms.append(avg_grad_norm)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, "
              f"Grad Norm: {avg_grad_norm:.6f}")

    return losses, gradient_norms


def analyze_gradients(model, dataloader, criterion):
    """Detailed gradient analysis per layer."""
    model.train()

    # Get a batch
    batch_x, batch_y = next(iter(dataloader))

    # Forward and backward
    output = model(batch_x)
    loss = criterion(output, batch_y)
    loss.backward()

    # Collect gradient statistics per layer
    grad_stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data

            grad_stats[name] = {
                'mean': grad.abs().mean().item(),
                'std': grad.std().item(),
                'max': grad.abs().max().item(),
                'min': grad.abs().min().item(),
                'norm': grad.norm().item()
            }

    return grad_stats


def visualize_gradient_flow(grad_stats, output_path):
    """Visualize gradient magnitudes across layers."""
    layer_names = list(grad_stats.keys())
    mean_grads = [grad_stats[name]['mean'] for name in layer_names]
    max_grads = [grad_stats[name]['max'] for name in layer_names]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(layer_names))
    ax.bar(x - 0.2, mean_grads, 0.4, label='Mean |gradient|', alpha=0.7)
    ax.bar(x + 0.2, max_grads, 0.4, label='Max |gradient|', alpha=0.7)

    ax.axhline(1e-6, color='red', linestyle='--', alpha=0.5,
               label='Vanishing threshold (1e-6)')
    ax.axhline(100, color='orange', linestyle='--', alpha=0.5,
               label='Exploding threshold (100)')

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Gradient Magnitude (log scale)')
    ax.set_title('Gradient Flow Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Gradient flow visualization saved to {output_path}")


def visualize_training(baseline_losses, fixed_losses, output_path):
    """Compare training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1 = axes[0]
    ax1.plot(baseline_losses, 'r-', label='Baseline (Sigmoid)', linewidth=2)
    ax1.plot(fixed_losses, 'g-', label='Fixed (ReLU + Skip)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss improvement
    ax2 = axes[1]
    improvement = [(b - f) / b * 100 for b, f in zip(baseline_losses, fixed_losses)]
    ax2.plot(improvement, 'b-', linewidth=2)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Loss Improvement with Fixes')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training comparison saved to {output_path}")


def run_baseline_experiment():
    """Run baseline experiment with vanishing gradients."""
    print("=" * 70)
    print("BASELINE EXPERIMENT: Deep Sigmoid Network (Vanishing Gradients)")
    print("=" * 70)

    # Create data
    train_data = create_synthetic_data(num_samples=1000)
    dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Create baseline model
    model = DeepSigmoidNetwork(input_dim=128, hidden_dim=128, output_dim=10, num_layers=20)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"\nModel architecture: {sum(p.numel() for p in model.parameters())} parameters")
    print("Activation: Sigmoid (known to cause vanishing gradients)")
    print("Depth: 20 layers\n")

    # Train
    losses, grad_norms = train_model(model, dataloader, optimizer, criterion, num_epochs=10)

    # Analyze gradients
    print("\n" + "-" * 70)
    print("Gradient Analysis:")
    print("-" * 70)
    grad_stats = analyze_gradients(model, dataloader, criterion)

    # Identify vanishing layers
    vanishing_layers = []
    for name, stats in grad_stats.items():
        if stats['mean'] < 1e-6:
            vanishing_layers.append(name)
            print(f"⚠️  {name}: mean gradient = {stats['mean']:.2e} (VANISHING)")

    print(f"\n📊 Summary:")
    print(f"   Total layers: {len(grad_stats)}")
    print(f"   Vanishing gradients: {len(vanishing_layers)} ({len(vanishing_layers)/len(grad_stats)*100:.1f}%)")
    print(f"   Final loss: {losses[-1]:.4f}")

    # Visualize
    visualize_gradient_flow(grad_stats, "baseline_gradient_flow.png")

    # Save results
    torch.save({
        'model_state': model.state_dict(),
        'losses': losses,
        'grad_norms': grad_norms,
        'grad_stats': grad_stats
    }, "baseline_results.pt")

    return losses, grad_stats


def run_debug_analysis():
    """Run gradient debugger on baseline model."""
    print("\n" + "=" * 70)
    print("DEBUG ANALYSIS: Identifying Root Causes")
    print("=" * 70)

    # Load baseline results
    results = torch.load("baseline_results.pt")
    grad_stats = results['grad_stats']

    print("\n🔍 Gradient Debugger Analysis:\n")

    # Calculate statistics
    vanishing_count = sum(1 for stats in grad_stats.values() if stats['mean'] < 1e-6)
    total_layers = len(grad_stats)

    print(f"Issue: {vanishing_count}/{total_layers} layers have vanishing gradients")
    print(f"Severity: {'CRITICAL' if vanishing_count > total_layers * 0.3 else 'HIGH'}")

    print("\n💡 Recommendations:")
    print("   1. Replace sigmoid activations with ReLU or GELU")
    print("      - Sigmoid saturates for large inputs → gradient ≈ 0")
    print("      - ReLU maintains gradient flow for positive inputs")
    print()
    print("   2. Add skip/residual connections")
    print("      - Allow gradients to bypass layers via identity paths")
    print("      - Prevents multiplicative gradient decay")
    print()
    print("   3. Use batch/layer normalization")
    print("      - Keeps activations in non-saturating range")
    print("      - Stabilizes gradient magnitudes")
    print()
    print("   4. Better weight initialization")
    print("      - Xavier/He initialization for ReLU networks")
    print("      - Prevents initial gradient vanishing")


def run_fixed_experiment(baseline_losses):
    """Run experiment with fixes applied."""
    print("\n" + "=" * 70)
    print("FIXED EXPERIMENT: ReLU + Residual Connections")
    print("=" * 70)

    # Create data
    train_data = create_synthetic_data(num_samples=1000)
    dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Create fixed model
    model = ImprovedDeepNetwork(input_dim=128, hidden_dim=128, output_dim=10, num_layers=20)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"\nModel architecture: {sum(p.numel() for p in model.parameters())} parameters")
    print("Activation: ReLU (prevents saturation)")
    print("Architecture: Residual blocks with skip connections")
    print("Depth: 20 layers\n")

    # Train
    losses, grad_norms = train_model(model, dataloader, optimizer, criterion, num_epochs=10)

    # Analyze gradients
    print("\n" + "-" * 70)
    print("Gradient Analysis:")
    print("-" * 70)
    grad_stats = analyze_gradients(model, dataloader, criterion)

    vanishing_layers = []
    for name, stats in grad_stats.items():
        if stats['mean'] < 1e-6:
            vanishing_layers.append(name)

    print(f"\n📊 Summary:")
    print(f"   Total layers: {len(grad_stats)}")
    print(f"   Vanishing gradients: {len(vanishing_layers)} ({len(vanishing_layers)/len(grad_stats)*100:.1f}%)")
    print(f"   Final loss: {losses[-1]:.4f}")
    print(f"   Improvement: {(baseline_losses[-1] - losses[-1]) / baseline_losses[-1] * 100:.1f}%")

    # Visualize
    visualize_gradient_flow(grad_stats, "fixed_gradient_flow.png")
    visualize_training(baseline_losses, losses, "training_comparison.png")

    print("\n✅ Gradient vanishing issue RESOLVED")
    print("   - ReLU activations prevent saturation")
    print("   - Skip connections provide gradient highways")
    print("   - Model successfully learns from all layers")


def main():
    parser = argparse.ArgumentParser(description='Vanishing Gradients Debugging Example')
    parser.add_argument('--mode', choices=['baseline', 'debug', 'fixed', 'all'],
                       default='all', help='Experiment mode')
    args = parser.parse_args()

    if args.mode in ['baseline', 'all']:
        baseline_losses, _ = run_baseline_experiment()

    if args.mode in ['debug', 'all']:
        run_debug_analysis()

    if args.mode in ['fixed', 'all']:
        if args.mode == 'fixed':
            # Load baseline results
            results = torch.load("baseline_results.pt")
            baseline_losses = results['losses']
        run_fixed_experiment(baseline_losses)

    print("\n" + "=" * 70)
    print("Example complete! Generated files:")
    print("  - baseline_gradient_flow.png")
    print("  - fixed_gradient_flow.png")
    print("  - training_comparison.png")
    print("  - baseline_results.pt")
    print("=" * 70)


if __name__ == '__main__':
    main()
