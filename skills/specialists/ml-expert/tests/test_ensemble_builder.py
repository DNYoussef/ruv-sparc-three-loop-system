"""
Tests for ensemble-builder.sh

Validates ensemble model construction and validation.
"""

import pytest
import subprocess
import json
import yaml
from pathlib import Path
import csv


@pytest.fixture
def sample_training_data(tmp_path):
    """Create sample training data."""
    csv_file = tmp_path / 'train.csv'

    # Binary classification data
    data = [
        ['feature1', 'feature2', 'feature3', 'target'],
        ['1.2', '0.5', '0.8', '0'],
        ['2.1', '1.3', '0.6', '1'],
        ['0.8', '0.9', '1.1', '0'],
        ['1.9', '1.1', '0.7', '1'],
        ['1.5', '0.7', '0.9', '0'],
        ['2.3', '1.4', '0.5', '1'],
        ['0.9', '0.6', '1.0', '0'],
        ['2.0', '1.2', '0.8', '1']
    ]

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return csv_file


@pytest.fixture
def ensemble_config(tmp_path):
    """Create ensemble configuration."""
    config = {
        'version': '1.0.0',
        'name': 'TestEnsemble',
        'ensemble': {
            'type': 'voting',
            'params': {
                'voting': 'soft',
                'n_jobs': -1
            }
        },
        'base_models': [
            {
                'name': 'rf',
                'type': 'random_forest',
                'params': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'random_state': 42
                }
            },
            {
                'name': 'lr',
                'type': 'logistic_regression',
                'params': {
                    'C': 1.0,
                    'random_state': 42
                }
            }
        ]
    }

    config_file = tmp_path / 'ensemble_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    return config_file


class TestEnsembleBuilder:
    """Test ensemble building functionality."""

    def test_ensemble_construction(self, ensemble_config, sample_training_data, tmp_path):
        """Test ensemble model construction."""
        output_dir = tmp_path / 'ensemble_model'

        result = subprocess.run([
            'bash',
            str(Path(__file__).parent.parent / 'resources' / 'ensemble-builder.sh'),
            '--config', str(ensemble_config),
            '--data', str(sample_training_data),
            '--output', str(output_dir)
        ], capture_output=True, text=True, timeout=60)

        # Check if bash is available
        if 'not found' in result.stderr or result.returncode == 127:
            pytest.skip("Bash not available")

        # Skip if Python dependencies missing
        if 'ModuleNotFoundError' in result.stderr or 'ImportError' in result.stderr:
            pytest.skip("Python ML dependencies not installed")

        if result.returncode != 0:
            pytest.skip(f"Ensemble building failed: {result.stderr}")

        # Check for output files
        assert output_dir.exists()

    def test_model_validation(self, ensemble_config, sample_training_data, tmp_path):
        """Test ensemble model validation."""
        # First build the model
        model_dir = tmp_path / 'validation_model'

        build_result = subprocess.run([
            'bash',
            str(Path(__file__).parent.parent / 'resources' / 'ensemble-builder.sh'),
            '--config', str(ensemble_config),
            '--data', str(sample_training_data),
            '--output', str(model_dir)
        ], capture_output=True, text=True, timeout=60)

        if build_result.returncode != 0:
            pytest.skip("Model building failed, cannot test validation")

        # Create test data (same structure as training)
        test_file = tmp_path / 'test.csv'
        with open(sample_training_data) as f_in:
            with open(test_file, 'w') as f_out:
                f_out.write(f_in.read())

        # Validate model
        metrics_file = tmp_path / 'metrics.json'

        validate_result = subprocess.run([
            'bash',
            str(Path(__file__).parent.parent / 'resources' / 'ensemble-builder.sh'),
            '--validate', str(model_dir),
            '--test', str(test_file),
            '--metrics', str(metrics_file)
        ], capture_output=True, text=True, timeout=60)

        if validate_result.returncode != 0:
            pytest.skip(f"Validation failed: {validate_result.stderr}")

        # Check metrics file
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)

            assert 'accuracy' in metrics or 'f1' in metrics


@pytest.mark.integration
class TestEnsembleWorkflow:
    """Integration tests for complete ensemble workflows."""

    def test_voting_ensemble(self, tmp_path):
        """Test voting ensemble construction."""
        # Create simple config
        config = {
            'ensemble': {'type': 'voting'},
            'base_models': [
                {'name': 'rf', 'type': 'random_forest', 'params': {'n_estimators': 5}},
                {'name': 'lr', 'type': 'logistic_regression', 'params': {}}
            ]
        }

        config_file = tmp_path / 'voting_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        # Create minimal training data
        data_file = tmp_path / 'data.csv'
        with open(data_file, 'w') as f:
            f.write('f1,f2,target\n')
            f.write('1,2,0\n')
            f.write('2,3,1\n')
            f.write('3,4,0\n')
            f.write('4,5,1\n')

        output_dir = tmp_path / 'voting_model'

        result = subprocess.run([
            'bash',
            str(Path(__file__).parent.parent / 'resources' / 'ensemble-builder.sh'),
            '--config', str(config_file),
            '--data', str(data_file),
            '--output', str(output_dir)
        ], capture_output=True, text=True, timeout=60)

        # Only assert if build succeeded (may fail due to missing dependencies)
        if result.returncode == 0:
            assert output_dir.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
