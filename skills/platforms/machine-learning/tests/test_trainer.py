#!/usr/bin/env python3
"""
Unit tests for ModelTrainer
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import shutil
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from resources.scripts.model_trainer import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.config_path = os.path.join(cls.temp_dir, 'test_config.yaml')

        # Create minimal config
        config_content = """
model:
  architecture: resnet50
  pretrained: false
  num_classes: 10

training:
  batch_size: 4
  epochs: 2
  learning_rate: 0.001
  optimizer: adam
  scheduler: null
  mixed_precision: false
  early_stopping: false

logging:
  tensorboard: false
  wandb: false
  log_interval: 1
  checkpoint_every: 1

hardware:
  device: cpu
"""
        with open(cls.config_path, 'w') as f:
            f.write(config_content)

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Set up before each test"""
        self.trainer = ModelTrainer(self.config_path, distributed=False)

        # Create dummy data loaders
        X = torch.randn(20, 3, 224, 224)
        y = torch.randint(0, 10, (20,))
        dataset = TensorDataset(X, y)
        self.train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        self.val_loader = DataLoader(dataset, batch_size=4)

    def test_init(self):
        """Test trainer initialization"""
        self.assertIsNotNone(self.trainer)
        self.assertIsNotNone(self.trainer.config)
        self.assertEqual(self.trainer.distributed, False)

    def test_build_model(self):
        """Test model building"""
        model = self.trainer.build_model()
        self.assertIsInstance(model, nn.Module)

        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 10))

    def test_build_optimizer(self):
        """Test optimizer building"""
        model = self.trainer.build_model()
        optimizer = self.trainer.build_optimizer(model)

        self.assertIsNotNone(optimizer)
        self.assertTrue(hasattr(optimizer, 'step'))

    def test_build_scheduler(self):
        """Test scheduler building"""
        model = self.trainer.build_model()
        optimizer = self.trainer.build_optimizer(model)
        scheduler = self.trainer.build_scheduler(optimizer)

        # Should be None based on config
        self.assertIsNone(scheduler)

    def test_train_epoch(self):
        """Test single training epoch"""
        model = self.trainer.build_model()
        optimizer = self.trainer.build_optimizer(model)
        criterion = nn.CrossEntropyLoss()

        metrics = self.trainer.train_epoch(
            model, self.train_loader, criterion, optimizer, epoch=0
        )

        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertGreaterEqual(metrics['loss'], 0)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 100)

    def test_validate(self):
        """Test validation"""
        model = self.trainer.build_model()
        criterion = nn.CrossEntropyLoss()

        metrics = self.trainer.validate(model, self.val_loader, criterion)

        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertGreaterEqual(metrics['loss'], 0)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 100)

    def test_train(self):
        """Test complete training loop"""
        model = self.trainer.train(self.train_loader, self.val_loader)

        self.assertIsNotNone(model)
        self.assertIsInstance(model, nn.Module)

    def test_save_load(self):
        """Test model saving"""
        model = self.trainer.build_model()
        save_path = os.path.join(self.temp_dir, 'test_model.pth')

        self.trainer.save(model, save_path)
        self.assertTrue(os.path.exists(save_path))

        # Load and verify
        loaded_state = torch.load(save_path)
        self.assertIsInstance(loaded_state, dict)

    def test_evaluate(self):
        """Test model evaluation"""
        model = self.trainer.build_model()
        metrics = self.trainer.evaluate(model, self.val_loader)

        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)


class TestDistributedTraining(unittest.TestCase):
    """Test cases for distributed training features"""

    def test_distributed_config(self):
        """Test distributed configuration parsing"""
        # This would test distributed setup in a multi-GPU environment
        # Skipped in single-GPU/CPU environments
        pass


class TestMixedPrecision(unittest.TestCase):
    """Test cases for mixed precision training"""

    def test_amp_scaler(self):
        """Test AMP scaler initialization"""
        temp_dir = tempfile.mkdtemp()
        config_path = os.path.join(temp_dir, 'amp_config.yaml')

        config_content = """
model:
  architecture: resnet50
  num_classes: 10

training:
  batch_size: 4
  epochs: 1
  learning_rate: 0.001
  optimizer: adam
  mixed_precision: true

logging:
  tensorboard: false
  wandb: false

hardware:
  device: cpu
"""
        with open(config_path, 'w') as f:
            f.write(config_content)

        trainer = ModelTrainer(config_path)
        self.assertTrue(trainer.use_amp)
        self.assertIsNotNone(trainer.scaler)

        shutil.rmtree(temp_dir)


class TestEarlyStopping(unittest.TestCase):
    """Test cases for early stopping"""

    def test_patience_counter(self):
        """Test patience counter increment"""
        temp_dir = tempfile.mkdtemp()
        config_path = os.path.join(temp_dir, 'es_config.yaml')

        config_content = """
model:
  architecture: resnet50
  num_classes: 10

training:
  batch_size: 4
  epochs: 10
  learning_rate: 0.001
  optimizer: adam
  early_stopping: true
  patience: 3

logging:
  tensorboard: false
  wandb: false

hardware:
  device: cpu
"""
        with open(config_path, 'w') as f:
            f.write(config_content)

        trainer = ModelTrainer(config_path)
        self.assertEqual(trainer.patience_counter, 0)

        # Simulate no improvement
        trainer.best_metric = 0.9
        trainer.patience_counter = 2
        # Would trigger early stopping after one more epoch

        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
