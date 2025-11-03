#!/usr/bin/env python3
"""
Test suite for ML Experiment Tracking

Tests experiment tracking functionality including:
- MLflow integration
- Parameter logging
- Metric logging
- Artifact management
- Experiment comparison
"""

import pytest
import tempfile
import shutil
import yaml
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'resources', 'scripts'))

# Mock mlflow if not installed
try:
    import mlflow
except ImportError:
    pytest.skip("MLflow not installed", allow_module_level=True)


class TestExperimentTracker:
    """Test cases for ExperimentTracker"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test artifacts"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration"""
        config = {
            'experiment': {
                'name': 'test-experiment',
                'tracking_uri': f'file://{temp_dir}/mlruns',
                'artifact_location': f'{temp_dir}/artifacts'
            },
            'training': {
                'epochs': 10,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }

        config_path = Path(temp_dir) / 'test-config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return str(config_path)

    def test_config_loading(self, test_config):
        """Test configuration loading from YAML"""
        from experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(test_config, backend='mlflow')
        assert tracker.config is not None
        assert tracker.config['experiment']['name'] == 'test-experiment'

    def test_mlflow_initialization(self, test_config):
        """Test MLflow tracking initialization"""
        from experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(test_config, backend='mlflow')
        assert mlflow.get_tracking_uri() is not None

    def test_start_run(self, test_config):
        """Test starting an experiment run"""
        from experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(test_config, backend='mlflow')
        run_id = tracker.start_run('test-run')

        assert run_id is not None
        assert tracker.run_id is not None

        tracker.end_run()

    def test_log_params(self, test_config):
        """Test logging parameters"""
        from experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(test_config, backend='mlflow')
        tracker.start_run('test-params')

        params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'optimizer': 'adam'
        }

        tracker.log_params(params)

        # Verify params were logged
        run = mlflow.get_run(tracker.run_id)
        assert run.data.params['learning_rate'] == '0.001'
        assert run.data.params['batch_size'] == '32'

        tracker.end_run()

    def test_log_metrics(self, test_config):
        """Test logging metrics"""
        from experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(test_config, backend='mlflow')
        tracker.start_run('test-metrics')

        metrics = {
            'accuracy': 0.95,
            'loss': 0.05
        }

        tracker.log_metrics(metrics, step=0)

        # Verify metrics were logged
        run = mlflow.get_run(tracker.run_id)
        assert 'accuracy' in run.data.metrics
        assert 'loss' in run.data.metrics

        tracker.end_run()

    def test_log_artifact(self, test_config, temp_dir):
        """Test logging artifacts"""
        from experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(test_config, backend='mlflow')
        tracker.start_run('test-artifacts')

        # Create test artifact
        artifact_path = Path(temp_dir) / 'test_artifact.txt'
        artifact_path.write_text('test content')

        tracker.log_artifact(str(artifact_path), 'test')

        # Verify artifact was logged
        run = mlflow.get_run(tracker.run_id)
        artifacts = mlflow.artifacts.list_artifacts(tracker.run_id)
        assert len(artifacts) > 0

        tracker.end_run()

    def test_multiple_metrics_over_time(self, test_config):
        """Test logging metrics over multiple steps"""
        from experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(test_config, backend='mlflow')
        tracker.start_run('test-multi-metrics')

        for step in range(10):
            metrics = {
                'train_loss': 1.0 / (step + 1),
                'val_loss': 1.2 / (step + 1),
                'accuracy': 0.5 + (step * 0.05)
            }
            tracker.log_metrics(metrics, step=step)

        # Verify metric history
        run = mlflow.get_run(tracker.run_id)
        metric_history = mlflow.tracking.MlflowClient().get_metric_history(
            tracker.run_id, 'train_loss'
        )
        assert len(metric_history) == 10

        tracker.end_run()

    def test_run_comparison(self, test_config):
        """Test comparing multiple runs"""
        from experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(test_config, backend='mlflow')

        # Create two runs
        run_ids = []
        for i in range(2):
            tracker.start_run(f'run-{i}')
            tracker.log_params({'learning_rate': 0.001 * (i + 1)})
            tracker.log_metrics({'accuracy': 0.8 + (i * 0.05)})
            run_ids.append(tracker.run_id)
            tracker.end_run()

        # Compare runs
        comparison = tracker.compare_runs(run_ids)

        assert len(comparison) == 2
        assert all(run_id in comparison for run_id in run_ids)

    def test_error_handling_invalid_config(self, temp_dir):
        """Test error handling for invalid configuration"""
        from experiment_tracker import ExperimentTracker

        invalid_config = Path(temp_dir) / 'invalid.yaml'
        invalid_config.write_text('invalid: yaml: content:')

        with pytest.raises(SystemExit):
            ExperimentTracker(str(invalid_config), backend='mlflow')

    def test_end_run_with_status(self, test_config):
        """Test ending run with different statuses"""
        from experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(test_config, backend='mlflow')

        # Test FINISHED status
        tracker.start_run('test-finished')
        tracker.end_run(status='FINISHED')

        run = mlflow.get_run(tracker.run_id)
        assert run.info.status == 'FINISHED'

        # Test FAILED status
        tracker.start_run('test-failed')
        tracker.end_run(status='FAILED')

        run = mlflow.get_run(tracker.run_id)
        assert run.info.status == 'FAILED'


class TestExperimentTrackerCLI:
    """Test CLI functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    def test_list_experiments(self, capsys):
        """Test listing experiments via CLI"""
        from experiment_tracker import list_experiments

        list_experiments()
        captured = capsys.readouterr()
        assert "Available Experiments" in captured.out

    def test_cli_integration(self, temp_dir):
        """Test CLI integration with configuration"""
        # Create minimal config
        config = {
            'experiment': {
                'name': 'cli-test',
                'tracking_uri': f'file://{temp_dir}/mlruns'
            },
            'training': {
                'epochs': 5
            }
        }

        config_path = Path(temp_dir) / 'cli-config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Test importing and basic usage
        from experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(str(config_path), backend='mlflow')
        assert tracker is not None


# Integration tests
class TestMLflowIntegration:
    """Integration tests with MLflow backend"""

    @pytest.fixture
    def mlflow_setup(self):
        """Setup MLflow tracking"""
        temp = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f'file://{temp}/mlruns')
        yield temp
        shutil.rmtree(temp)

    def test_full_experiment_workflow(self, mlflow_setup):
        """Test complete experiment workflow"""
        from experiment_tracker import ExperimentTracker

        # Create config
        config_path = Path(mlflow_setup) / 'config.yaml'
        config = {
            'experiment': {
                'name': 'integration-test',
                'tracking_uri': f'file://{mlflow_setup}/mlruns'
            },
            'training': {
                'epochs': 10,
                'batch_size': 32
            }
        }

        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Run full workflow
        tracker = ExperimentTracker(str(config_path), backend='mlflow')
        tracker.start_run('integration-run')

        # Log params
        tracker.log_params(config['training'])

        # Simulate training loop
        for epoch in range(5):
            tracker.log_metrics({
                'train_loss': 1.0 / (epoch + 1),
                'val_accuracy': 0.7 + (epoch * 0.05)
            }, step=epoch)

        tracker.end_run('FINISHED')

        # Verify complete workflow
        assert tracker.run_id is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
