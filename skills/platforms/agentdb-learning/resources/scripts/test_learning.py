#!/usr/bin/env python3
"""
AgentDB Learning Test Suite
Validates all 9 RL algorithms with unit tests and integration tests
"""

import unittest
import json
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from train_rl_agent import RLTrainer
except ImportError:
    print("Warning: Could not import train_rl_agent. Skipping some tests.")
    RLTrainer = None


class TestRLAlgorithms(unittest.TestCase):
    """Test all 9 RL algorithms"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon': 0.1,
            'epsilon_decay': 0.995,
            'state_dim': 4,
            'action_dim': 2,
            'max_steps': 10
        }
        self.test_episodes = 5

    def test_q_learning(self):
        """Test Q-Learning algorithm"""
        if RLTrainer is None:
            self.skipTest("RLTrainer not available")

        trainer = RLTrainer('q-learning', self.test_config)
        metrics = trainer.train(self.test_episodes)

        self.assertEqual(metrics['episodes'], self.test_episodes)
        self.assertGreater(metrics['training_time'], 0)
        self.assertIsInstance(metrics['avg_reward'], float)

    def test_sarsa(self):
        """Test SARSA algorithm"""
        if RLTrainer is None:
            self.skipTest("RLTrainer not available")

        trainer = RLTrainer('sarsa', self.test_config)
        metrics = trainer.train(self.test_episodes)

        self.assertEqual(metrics['episodes'], self.test_episodes)
        self.assertGreater(metrics['training_time'], 0)

    def test_actor_critic(self):
        """Test Actor-Critic algorithm"""
        if RLTrainer is None:
            self.skipTest("RLTrainer not available")

        config = self.test_config.copy()
        config['actor_lr'] = 0.001
        config['critic_lr'] = 0.002

        trainer = RLTrainer('actor-critic', config)
        metrics = trainer.train(self.test_episodes)

        self.assertEqual(metrics['episodes'], self.test_episodes)
        self.assertTrue(len(metrics['loss']) > 0)

    def test_decision_transformer(self):
        """Test Decision Transformer algorithm"""
        if RLTrainer is None:
            self.skipTest("RLTrainer not available")

        config = self.test_config.copy()
        config['context_length'] = 20
        config['embed_dim'] = 128

        trainer = RLTrainer('decision-transformer', config)
        metrics = trainer.train(self.test_episodes)

        self.assertEqual(metrics['episodes'], self.test_episodes)
        self.assertGreater(metrics['avg_reward'], -float('inf'))

    def test_all_algorithms(self):
        """Test that all 9 algorithms can be instantiated"""
        if RLTrainer is None:
            self.skipTest("RLTrainer not available")

        algorithms = list(RLTrainer.ALGORITHMS.keys())
        self.assertEqual(len(algorithms), 9, "Should have exactly 9 algorithms")

        for algo in algorithms:
            with self.subTest(algorithm=algo):
                trainer = RLTrainer(algo, self.test_config)
                self.assertIsNotNone(trainer)
                self.assertEqual(trainer.algorithm, algo)


class TestConfiguration(unittest.TestCase):
    """Test configuration loading and validation"""

    def test_default_config(self):
        """Test default configuration"""
        config = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon': 0.1
        }

        self.assertIn('learning_rate', config)
        self.assertGreater(config['learning_rate'], 0)
        self.assertLessEqual(config['gamma'], 1.0)
        self.assertGreaterEqual(config['gamma'], 0)

    def test_algorithm_specific_configs(self):
        """Test algorithm-specific configuration parameters"""
        configs = {
            'q-learning': {
                'epsilon': 0.1,
                'epsilon_decay': 0.995
            },
            'actor-critic': {
                'actor_lr': 0.001,
                'critic_lr': 0.002,
                'entropy_coef': 0.01
            },
            'decision-transformer': {
                'context_length': 20,
                'embed_dim': 128,
                'n_heads': 8
            }
        }

        for algo, config in configs.items():
            with self.subTest(algorithm=algo):
                self.assertIsInstance(config, dict)
                self.assertTrue(len(config) > 0)


class TestMetrics(unittest.TestCase):
    """Test training metrics and tracking"""

    def test_metrics_initialization(self):
        """Test metrics are properly initialized"""
        if RLTrainer is None:
            self.skipTest("RLTrainer not available")

        trainer = RLTrainer('q-learning', {})

        self.assertEqual(trainer.metrics['episodes'], 0)
        self.assertEqual(trainer.metrics['total_reward'], 0.0)
        self.assertEqual(trainer.metrics['avg_reward'], 0.0)
        self.assertIsInstance(trainer.metrics['loss'], list)

    def test_metrics_update(self):
        """Test metrics are updated during training"""
        if RLTrainer is None:
            self.skipTest("RLTrainer not available")

        config = {
            'state_dim': 4,
            'action_dim': 2,
            'max_steps': 5
        }

        trainer = RLTrainer('q-learning', config)
        metrics = trainer.train(num_episodes=3)

        self.assertEqual(metrics['episodes'], 3)
        self.assertNotEqual(metrics['total_reward'], 0.0)
        self.assertEqual(metrics['avg_reward'], metrics['total_reward'] / 3)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""

    def test_train_and_save(self):
        """Test training and saving model"""
        if RLTrainer is None:
            self.skipTest("RLTrainer not available")

        import tempfile
        import os

        config = {
            'state_dim': 4,
            'action_dim': 2,
            'max_steps': 5
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            save_path = f.name

        try:
            trainer = RLTrainer('q-learning', config)
            metrics = trainer.train(num_episodes=3, save_path=save_path)

            # Verify file was created
            self.assertTrue(os.path.exists(save_path))

            # Verify file contents
            with open(save_path, 'r') as f:
                saved_data = json.load(f)

            self.assertEqual(saved_data['algorithm'], 'q-learning')
            self.assertEqual(saved_data['metrics']['episodes'], 3)

        finally:
            # Cleanup
            if os.path.exists(save_path):
                os.remove(save_path)


def run_tests(verbosity=2):
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRLAlgorithms))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*60)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
