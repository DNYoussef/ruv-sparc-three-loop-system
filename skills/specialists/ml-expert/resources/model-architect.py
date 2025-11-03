#!/usr/bin/env python3
"""
Neural Architecture Design and Construction Tool

This script provides utilities for designing, building, and validating
neural network architectures with best practices for modularity,
efficiency, and maintainability.

Usage:
    python model-architect.py --config architecture-template.yaml --output model.py
    python model-architect.py --validate existing_model.py
    python model-architect.py --optimize model.py --target-params 25M
"""

import argparse
import yaml
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LayerConfig:
    """Configuration for a single neural network layer."""
    type: str  # 'linear', 'conv2d', 'attention', 'lstm', etc.
    input_dim: int
    output_dim: int
    activation: Optional[str] = 'relu'
    dropout: float = 0.0
    batch_norm: bool = False
    params: Dict = field(default_factory=dict)


@dataclass
class ArchitectureConfig:
    """Complete neural network architecture specification."""
    name: str
    layers: List[LayerConfig]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    target_params: Optional[int] = None
    constraints: Dict = field(default_factory=dict)


class ModelArchitect:
    """Neural network architecture designer and builder."""

    LAYER_REGISTRY = {
        'linear': nn.Linear,
        'conv2d': nn.Conv2d,
        'conv1d': nn.Conv1d,
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'embedding': nn.Embedding,
        'layernorm': nn.LayerNorm,
        'batchnorm1d': nn.BatchNorm1d,
        'batchnorm2d': nn.BatchNorm2d,
        'dropout': nn.Dropout,
    }

    ACTIVATION_REGISTRY = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'softmax': nn.Softmax,
        'leaky_relu': nn.LeakyReLU,
    }

    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.model = None

    def build(self) -> nn.Module:
        """Build the neural network from configuration."""
        layers = []

        for i, layer_config in enumerate(self.config.layers):
            # Build main layer
            layer = self._build_layer(layer_config)
            layers.append((f"{layer_config.type}_{i}", layer))

            # Add batch normalization if specified
            if layer_config.batch_norm:
                bn = self._build_batch_norm(layer_config)
                layers.append((f"bn_{i}", bn))

            # Add activation function
            if layer_config.activation:
                activation = self._build_activation(layer_config.activation)
                layers.append((f"{layer_config.activation}_{i}", activation))

            # Add dropout if specified
            if layer_config.dropout > 0:
                dropout = nn.Dropout(p=layer_config.dropout)
                layers.append((f"dropout_{i}", dropout))

        self.model = nn.Sequential(dict(layers))
        return self.model

    def _build_layer(self, config: LayerConfig) -> nn.Module:
        """Build a single layer from configuration."""
        layer_class = self.LAYER_REGISTRY.get(config.type)
        if not layer_class:
            raise ValueError(f"Unknown layer type: {config.type}")

        # Build layer with specific parameters
        if config.type in ['linear']:
            return layer_class(config.input_dim, config.output_dim)
        elif config.type in ['conv2d']:
            return layer_class(
                config.input_dim,
                config.output_dim,
                kernel_size=config.params.get('kernel_size', 3),
                stride=config.params.get('stride', 1),
                padding=config.params.get('padding', 1)
            )
        elif config.type in ['lstm', 'gru']:
            return layer_class(
                config.input_dim,
                config.output_dim,
                num_layers=config.params.get('num_layers', 1),
                batch_first=True,
                dropout=config.params.get('dropout', 0.0)
            )
        else:
            return layer_class(**config.params)

    def _build_batch_norm(self, config: LayerConfig) -> nn.Module:
        """Build batch normalization layer."""
        if 'conv' in config.type:
            return nn.BatchNorm2d(config.output_dim)
        else:
            return nn.BatchNorm1d(config.output_dim)

    def _build_activation(self, activation: str) -> nn.Module:
        """Build activation function."""
        activation_class = self.ACTIVATION_REGISTRY.get(activation)
        if not activation_class:
            raise ValueError(f"Unknown activation: {activation}")
        return activation_class()

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        if self.model is None:
            self.build()
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def validate_constraints(self) -> Dict[str, bool]:
        """Validate architecture against constraints."""
        results = {}

        # Check parameter count
        if self.config.target_params:
            param_count = self.count_parameters()
            target = self.config.target_params
            tolerance = self.config.constraints.get('param_tolerance', 0.1)
            within_tolerance = abs(param_count - target) / target <= tolerance
            results['param_count'] = within_tolerance

        # Check memory constraints
        if 'max_memory_mb' in self.config.constraints:
            memory_mb = self.estimate_memory()
            results['memory'] = memory_mb <= self.config.constraints['max_memory_mb']

        # Check inference speed constraints
        if 'max_inference_ms' in self.config.constraints:
            inference_ms = self.benchmark_inference()
            results['inference_speed'] = inference_ms <= self.config.constraints['max_inference_ms']

        return results

    def estimate_memory(self) -> float:
        """Estimate memory usage in MB."""
        if self.model is None:
            self.build()

        # Parameters
        param_memory = sum(p.numel() * p.element_size() for p in self.model.parameters())

        # Activations (rough estimate)
        activation_memory = param_memory * 2  # Conservative estimate

        total_bytes = param_memory + activation_memory
        return total_bytes / (1024 ** 2)  # Convert to MB

    def benchmark_inference(self, num_samples: int = 100) -> float:
        """Benchmark inference speed in milliseconds."""
        if self.model is None:
            self.build()

        self.model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Create dummy input
        dummy_input = torch.randn(1, *self.config.input_shape).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)

        # Benchmark
        import time
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_samples):
                _ = self.model(dummy_input)
        end = time.perf_counter()

        avg_time_ms = ((end - start) / num_samples) * 1000
        return avg_time_ms

    def generate_code(self, output_path: Path) -> None:
        """Generate PyTorch model code from architecture."""
        code = self._generate_model_class()

        with open(output_path, 'w') as f:
            f.write(code)

    def _generate_model_class(self) -> str:
        """Generate complete model class code."""
        layers_code = []
        forward_code = []

        for i, layer_config in enumerate(self.config.layers):
            # Generate layer initialization
            layer_init = self._generate_layer_init(layer_config, i)
            layers_code.append(layer_init)

            # Generate forward pass
            forward_code.append(f"        x = self.layer_{i}(x)")

            if layer_config.batch_norm:
                forward_code.append(f"        x = self.bn_{i}(x)")

            if layer_config.activation:
                forward_code.append(f"        x = self.{layer_config.activation}_{i}(x)")

            if layer_config.dropout > 0:
                forward_code.append(f"        x = self.dropout_{i}(x)")

        # Build complete class
        template = f'''import torch
import torch.nn as nn

class {self.config.name}(nn.Module):
    """
    {self.config.name} - Auto-generated by ModelArchitect

    Input shape: {self.config.input_shape}
    Output shape: {self.config.output_shape}
    Parameters: {self.count_parameters():,}
    """

    def __init__(self):
        super().__init__()

{chr(10).join(layers_code)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
{chr(10).join(forward_code)}
        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
'''
        return template

    def _generate_layer_init(self, config: LayerConfig, index: int) -> str:
        """Generate initialization code for a layer."""
        lines = []

        # Main layer
        if config.type == 'linear':
            lines.append(f"        self.layer_{index} = nn.Linear({config.input_dim}, {config.output_dim})")
        elif config.type == 'conv2d':
            k = config.params.get('kernel_size', 3)
            s = config.params.get('stride', 1)
            p = config.params.get('padding', 1)
            lines.append(f"        self.layer_{index} = nn.Conv2d({config.input_dim}, {config.output_dim}, kernel_size={k}, stride={s}, padding={p})")

        # Batch norm
        if config.batch_norm:
            if 'conv' in config.type:
                lines.append(f"        self.bn_{index} = nn.BatchNorm2d({config.output_dim})")
            else:
                lines.append(f"        self.bn_{index} = nn.BatchNorm1d({config.output_dim})")

        # Activation
        if config.activation:
            act_class = self.ACTIVATION_REGISTRY[config.activation].__name__
            lines.append(f"        self.{config.activation}_{index} = nn.{act_class}()")

        # Dropout
        if config.dropout > 0:
            lines.append(f"        self.dropout_{index} = nn.Dropout(p={config.dropout})")

        return '\n'.join(lines)


def load_config(config_path: Path) -> ArchitectureConfig:
    """Load architecture configuration from YAML."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    layers = [LayerConfig(**layer) for layer in data['layers']]

    return ArchitectureConfig(
        name=data['name'],
        layers=layers,
        input_shape=tuple(data['input_shape']),
        output_shape=tuple(data['output_shape']),
        target_params=data.get('target_params'),
        constraints=data.get('constraints', {})
    )


def main():
    parser = argparse.ArgumentParser(description='Neural Architecture Designer')
    parser.add_argument('--config', type=Path, help='Architecture configuration YAML')
    parser.add_argument('--output', type=Path, help='Output model file')
    parser.add_argument('--validate', type=Path, help='Validate existing model')
    parser.add_argument('--optimize', type=Path, help='Optimize architecture')
    parser.add_argument('--target-params', type=str, help='Target parameter count (e.g., 25M)')

    args = parser.parse_args()

    if args.config:
        # Build model from config
        config = load_config(args.config)
        architect = ModelArchitect(config)

        print(f"Building {config.name}...")
        model = architect.build()
        print(f"Parameters: {architect.count_parameters():,}")
        print(f"Estimated memory: {architect.estimate_memory():.2f} MB")

        # Validate constraints
        validation = architect.validate_constraints()
        print(f"Validation: {validation}")

        # Generate code if output specified
        if args.output:
            architect.generate_code(args.output)
            print(f"Generated model code: {args.output}")

    elif args.validate:
        print(f"Validating {args.validate}...")
        # TODO: Implement model validation

    elif args.optimize:
        print(f"Optimizing {args.optimize}...")
        # TODO: Implement architecture optimization


if __name__ == '__main__':
    main()
