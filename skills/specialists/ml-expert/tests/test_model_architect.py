"""
Tests for model-architect.py

Validates neural architecture design, construction, and code generation.
"""

import pytest
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import sys

# Add resources to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'resources'))

try:
    from model_architect import (
        ModelArchitect, LayerConfig, ArchitectureConfig, load_config
    )
    ARCHITECT_AVAILABLE = True
except ImportError:
    ARCHITECT_AVAILABLE = False


@pytest.fixture
def simple_layer_config():
    """Fixture for a simple layer configuration."""
    return LayerConfig(
        type='linear',
        input_dim=784,
        output_dim=128,
        activation='relu',
        dropout=0.2,
        batch_norm=True
    )


@pytest.fixture
def simple_architecture_config():
    """Fixture for a simple architecture configuration."""
    layers = [
        LayerConfig(type='linear', input_dim=784, output_dim=512, activation='relu'),
        LayerConfig(type='linear', input_dim=512, output_dim=256, activation='relu'),
        LayerConfig(type='linear', input_dim=256, output_dim=10, activation='softmax')
    ]

    return ArchitectureConfig(
        name='SimpleNN',
        layers=layers,
        input_shape=(784,),
        output_shape=(10,),
        target_params=500000
    )


@pytest.mark.skipif(not ARCHITECT_AVAILABLE, reason="model_architect not available")
class TestModelArchitect:
    """Test ModelArchitect class functionality."""

    def test_layer_construction(self, simple_layer_config):
        """Test individual layer construction."""
        config = ArchitectureConfig(
            name='TestModel',
            layers=[simple_layer_config],
            input_shape=(784,),
            output_shape=(128,)
        )

        architect = ModelArchitect(config)
        layer = architect._build_layer(simple_layer_config)

        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 784
        assert layer.out_features == 128

    def test_activation_construction(self):
        """Test activation function construction."""
        config = ArchitectureConfig(
            name='TestModel',
            layers=[],
            input_shape=(784,),
            output_shape=(10,)
        )

        architect = ModelArchitect(config)

        relu = architect._build_activation('relu')
        assert isinstance(relu, nn.ReLU)

        gelu = architect._build_activation('gelu')
        assert isinstance(gelu, nn.GELU)

    def test_model_building(self, simple_architecture_config):
        """Test complete model building."""
        architect = ModelArchitect(simple_architecture_config)
        model = architect.build()

        assert isinstance(model, nn.Sequential)
        assert len(model) > 0

    def test_parameter_counting(self, simple_architecture_config):
        """Test parameter counting."""
        architect = ModelArchitect(simple_architecture_config)
        architect.build()

        param_count = architect.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)

    def test_forward_pass(self, simple_architecture_config):
        """Test that built model can perform forward pass."""
        architect = ModelArchitect(simple_architecture_config)
        model = architect.build()

        # Create dummy input
        batch_size = 4
        dummy_input = torch.randn(batch_size, 784)

        # Forward pass
        output = model(dummy_input)

        assert output.shape == (batch_size, 10)

    def test_batch_norm_addition(self):
        """Test batch normalization layer addition."""
        layer_config = LayerConfig(
            type='linear',
            input_dim=100,
            output_dim=50,
            activation='relu',
            batch_norm=True
        )

        config = ArchitectureConfig(
            name='BNTest',
            layers=[layer_config],
            input_shape=(100,),
            output_shape=(50,)
        )

        architect = ModelArchitect(config)
        model = architect.build()

        # Check for batch norm layer
        has_bn = any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
        assert has_bn

    def test_dropout_addition(self):
        """Test dropout layer addition."""
        layer_config = LayerConfig(
            type='linear',
            input_dim=100,
            output_dim=50,
            activation='relu',
            dropout=0.5
        )

        config = ArchitectureConfig(
            name='DropoutTest',
            layers=[layer_config],
            input_shape=(100,),
            output_shape=(50,)
        )

        architect = ModelArchitect(config)
        model = architect.build()

        # Check for dropout layer
        has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
        assert has_dropout

    def test_memory_estimation(self, simple_architecture_config):
        """Test memory usage estimation."""
        architect = ModelArchitect(simple_architecture_config)
        architect.build()

        memory_mb = architect.estimate_memory()

        assert memory_mb > 0
        assert isinstance(memory_mb, float)

    def test_constraint_validation(self):
        """Test architecture constraint validation."""
        layers = [
            LayerConfig(type='linear', input_dim=100, output_dim=50, activation='relu')
        ]

        config = ArchitectureConfig(
            name='ConstraintTest',
            layers=layers,
            input_shape=(100,),
            output_shape=(50,),
            target_params=5000,
            constraints={'param_tolerance': 0.2}
        )

        architect = ModelArchitect(config)
        architect.build()

        validation = architect.validate_constraints()
        assert 'param_count' in validation
        assert isinstance(validation['param_count'], bool)

    def test_code_generation(self, simple_architecture_config, tmp_path):
        """Test PyTorch code generation."""
        architect = ModelArchitect(simple_architecture_config)
        architect.build()

        output_file = tmp_path / 'generated_model.py'
        architect.generate_code(output_file)

        assert output_file.exists()

        # Check generated code contains key elements
        code = output_file.read_text()
        assert 'import torch' in code
        assert 'class SimpleNN' in code
        assert 'def forward' in code


@pytest.mark.skipif(not ARCHITECT_AVAILABLE, reason="model_architect not available")
class TestConfigLoading:
    """Test configuration loading functionality."""

    def test_load_yaml_config(self, tmp_path):
        """Test loading architecture from YAML config."""
        config_data = {
            'name': 'TestNetwork',
            'input_shape': [784],
            'output_shape': [10],
            'layers': [
                {
                    'type': 'linear',
                    'input_dim': 784,
                    'output_dim': 128,
                    'activation': 'relu'
                }
            ]
        }

        config_file = tmp_path / 'test_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)

        assert config.name == 'TestNetwork'
        assert config.input_shape == (784,)
        assert config.output_shape == (10,)
        assert len(config.layers) == 1


@pytest.mark.integration
@pytest.mark.skipif(not ARCHITECT_AVAILABLE, reason="model_architect not available")
class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow from config to trained model."""
        # Create config
        config_data = {
            'name': 'E2ETest',
            'input_shape': [10],
            'output_shape': [2],
            'target_params': 1000,
            'layers': [
                {'type': 'linear', 'input_dim': 10, 'output_dim': 20, 'activation': 'relu'},
                {'type': 'linear', 'input_dim': 20, 'output_dim': 2, 'activation': 'softmax'}
            ]
        }

        config_file = tmp_path / 'e2e_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        # Load and build
        config = load_config(config_file)
        architect = ModelArchitect(config)
        model = architect.build()

        # Test forward pass
        dummy_input = torch.randn(5, 10)
        output = model(dummy_input)

        assert output.shape == (5, 2)

        # Generate code
        output_file = tmp_path / 'e2e_model.py'
        architect.generate_code(output_file)

        assert output_file.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
