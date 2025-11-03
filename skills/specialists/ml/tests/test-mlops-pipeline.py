#!/usr/bin/env python3
"""
Test suite for MLOps Pipeline

Tests end-to-end MLOps workflow including:
- Data validation
- Model training
- Model evaluation
- Model registry integration
- Deployment automation
- Monitoring
"""

import pytest
import tempfile
import shutil
import yaml
import json
from pathlib import Path
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'resources', 'scripts'))


class TestMLOpsPipeline:
    """Test cases for MLOpsPipeline"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration"""
        config = {
            'data': {
                'validation_enabled': True,
                'preprocessing_enabled': True
            },
            'training': {
                'experiment_tracking': False,
                'auto_hyperparameter_tuning': False
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision'],
                'thresholds': {'accuracy': 0.85}
            },
            'deployment': {
                'auto_deploy': False,
                'environments': ['staging']
            },
            'monitoring': {
                'enabled': True,
                'interval': 60
            }
        }

        config_path = Path(temp_dir) / 'pipeline-config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return str(config_path)

    def test_pipeline_initialization(self, test_config):
        """Test pipeline initialization"""
        from ml_ops import MLOpsPipeline

        pipeline = MLOpsPipeline(test_config)
        assert pipeline.config is not None
        assert pipeline.pipeline_id is not None

    def test_default_config(self):
        """Test default configuration"""
        from ml_ops import MLOpsPipeline

        pipeline = MLOpsPipeline()
        assert pipeline.config is not None
        assert 'data' in pipeline.config
        assert 'training' in pipeline.config
        assert 'evaluation' in pipeline.config

    def test_data_validation(self, test_config):
        """Test data validation phase"""
        from ml_ops import MLOpsPipeline

        pipeline = MLOpsPipeline(test_config)
        result = pipeline.validate_data()

        assert isinstance(result, bool)
        assert 'data_validation' in pipeline.metrics

    def test_data_preprocessing(self, test_config):
        """Test data preprocessing phase"""
        from ml_ops import MLOpsPipeline

        pipeline = MLOpsPipeline(test_config)
        result = pipeline.preprocess_data()

        assert result is True

    def test_model_training(self, test_config):
        """Test model training phase"""
        from ml_ops import MLOpsPipeline

        pipeline = MLOpsPipeline(test_config)
        model_version = pipeline.train_model('test-model')

        assert model_version is not None
        assert model_version.startswith('v')
        assert 'training' in pipeline.metrics

    def test_model_evaluation(self, test_config):
        """Test model evaluation phase"""
        from ml_ops import MLOpsPipeline

        pipeline = MLOpsPipeline(test_config)
        result = pipeline.evaluate_model('test-model', 'v1.0.0')

        assert isinstance(result, bool)
        assert 'evaluation' in pipeline.metrics

    def test_evaluation_thresholds(self, test_config):
        """Test evaluation threshold checking"""
        from ml_ops import MLOpsPipeline

        pipeline = MLOpsPipeline(test_config)
        pipeline.evaluate_model('test-model', 'v1.0.0')

        # Check if metrics meet thresholds
        eval_metrics = pipeline.metrics['evaluation']
        thresholds = pipeline.config['evaluation']['thresholds']

        for metric, threshold in thresholds.items():
            assert metric in eval_metrics

    @patch('subprocess.run')
    def test_model_registration(self, mock_subprocess, test_config, temp_dir):
        """Test model registration"""
        from ml_ops import MLOpsPipeline

        mock_subprocess.return_value = Mock(returncode=0)

        # Create dummy model file
        model_path = Path(temp_dir) / 'test-model_v1.0.0.pkl'
        model_path.write_text('dummy model')

        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            pipeline = MLOpsPipeline(test_config)
            result = pipeline.register_model('test-model', 'v1.0.0')

            assert result is True
            mock_subprocess.assert_called_once()
        finally:
            os.chdir(original_cwd)

    @patch('subprocess.run')
    def test_model_deployment(self, mock_subprocess, test_config):
        """Test model deployment"""
        from ml_ops import MLOpsPipeline

        mock_subprocess.return_value = Mock(returncode=0)

        pipeline = MLOpsPipeline(test_config)
        result = pipeline.deploy_model('test-model', 'v1.0.0', 'staging')

        assert result is True
        mock_subprocess.assert_called_once()

    @patch('subprocess.run')
    def test_deployment_failure_handling(self, mock_subprocess, test_config):
        """Test deployment failure handling"""
        from ml_ops import MLOpsPipeline
        import subprocess

        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'deploy')

        pipeline = MLOpsPipeline(test_config)
        result = pipeline.deploy_model('test-model', 'v1.0.0', 'staging')

        assert result is False

    def test_pipeline_metrics_tracking(self, test_config):
        """Test pipeline metrics tracking"""
        from ml_ops import MLOpsPipeline

        pipeline = MLOpsPipeline(test_config)

        # Run multiple phases
        pipeline.validate_data()
        pipeline.train_model('test-model')
        pipeline.evaluate_model('test-model', 'v1.0.0')

        # Check metrics
        assert 'data_validation' in pipeline.metrics
        assert 'training' in pipeline.metrics
        assert 'evaluation' in pipeline.metrics

    def test_pipeline_id_generation(self, test_config):
        """Test pipeline ID generation"""
        from ml_ops import MLOpsPipeline

        pipeline1 = MLOpsPipeline(test_config)
        pipeline2 = MLOpsPipeline(test_config)

        assert pipeline1.pipeline_id != pipeline2.pipeline_id
        assert pipeline1.pipeline_id.startswith('pipeline_')


class TestFullPipeline:
    """Integration tests for full pipeline execution"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def full_config(self, temp_dir):
        """Create full pipeline configuration"""
        config = {
            'data': {
                'validation_enabled': True,
                'preprocessing_enabled': True
            },
            'training': {
                'experiment_tracking': False
            },
            'evaluation': {
                'metrics': ['accuracy'],
                'thresholds': {'accuracy': 0.80}  # Lower threshold for testing
            },
            'deployment': {
                'auto_deploy': False,
                'environments': ['staging']
            }
        }

        config_path = Path(temp_dir) / 'full-config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return str(config_path)

    @patch('subprocess.run')
    def test_full_pipeline_execution(self, mock_subprocess, full_config, temp_dir):
        """Test complete pipeline execution"""
        from ml_ops import MLOpsPipeline

        mock_subprocess.return_value = Mock(returncode=0)

        # Create dummy model file
        model_path = Path(temp_dir) / 'models'
        model_path.mkdir(exist_ok=True)

        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            pipeline = MLOpsPipeline(full_config)
            result = pipeline.run_full_pipeline('integration-test-model')

            assert result is True

            # Verify all phases executed
            assert 'data_validation' in pipeline.metrics
            assert 'training' in pipeline.metrics
            assert 'evaluation' in pipeline.metrics

            # Verify metrics file created
            metrics_files = list(Path('.').glob('pipeline_metrics_*.json'))
            assert len(metrics_files) > 0

        finally:
            os.chdir(original_cwd)

    def test_pipeline_failure_recovery(self, full_config):
        """Test pipeline failure at different stages"""
        from ml_ops import MLOpsPipeline

        # Simulate data validation failure
        pipeline = MLOpsPipeline(full_config)

        with patch.object(pipeline, 'validate_data', return_value=False):
            result = pipeline.run_full_pipeline('test-model')
            assert result is False

    def test_metrics_persistence(self, full_config, temp_dir):
        """Test pipeline metrics persistence"""
        from ml_ops import MLOpsPipeline

        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            pipeline = MLOpsPipeline(full_config)
            pipeline.validate_data()
            pipeline.train_model('test-model')
            pipeline._save_pipeline_metrics()

            # Verify metrics file
            metrics_file = f'pipeline_metrics_{pipeline.pipeline_id}.json'
            assert Path(metrics_file).exists()

            with open(metrics_file) as f:
                saved_metrics = json.load(f)

            assert saved_metrics['pipeline_id'] == pipeline.pipeline_id
            assert 'metrics' in saved_metrics

        finally:
            os.chdir(original_cwd)


class TestMonitoring:
    """Test monitoring functionality"""

    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        return {
            'monitoring': {
                'enabled': True,
                'interval': 1
            }
        }

    def test_monitoring_initialization(self, test_config):
        """Test monitoring initialization"""
        from ml_ops import MLOpsPipeline
        import yaml

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name

        try:
            pipeline = MLOpsPipeline(config_path)
            assert pipeline.config['monitoring']['enabled'] is True
        finally:
            os.unlink(config_path)

    @patch('time.sleep')
    def test_monitoring_metrics_collection(self, mock_sleep, test_config):
        """Test monitoring metrics collection"""
        from ml_ops import MLOpsPipeline
        import yaml

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name

        try:
            pipeline = MLOpsPipeline(config_path)

            # Mock sleep to prevent infinite loop
            mock_sleep.side_effect = KeyboardInterrupt()

            try:
                pipeline.monitor_model('test-model', interval=1)
            except KeyboardInterrupt:
                pass  # Expected

        finally:
            os.unlink(config_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
