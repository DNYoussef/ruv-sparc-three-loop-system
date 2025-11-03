#!/usr/bin/env python3
"""
Test suite for loss divergence detection

Tests the loss-analyzer.py script's ability to detect and diagnose
loss divergence patterns in ML training.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import the analyzer
sys.path.insert(0, str(Path(__file__).parent.parent / 'resources' / 'scripts'))

try:
    from loss_analyzer import LossAnalyzer
except ImportError:
    # Create a mock for testing without dependencies
    class LossAnalyzer:
        def __init__(self, **kwargs):
            pass


class TestLossDivergence(unittest.TestCase):
    """Test cases for loss divergence detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = LossAnalyzer(window_size=5, divergence_threshold=0.15)

    def test_no_divergence(self):
        """Test that healthy decreasing loss is not flagged."""
        # Simulate healthy exponential decay
        losses = 10.0 * np.exp(-0.1 * np.arange(50))
        result = self.analyzer.detect_divergence(losses)

        self.assertEqual(result['count'], 0)
        self.assertEqual(result['severity'], 'none')
        self.assertEqual(len(result['points']), 0)

    def test_sudden_divergence(self):
        """Test detection of sudden loss spike."""
        # Healthy decrease then sudden spike
        losses = np.concatenate([
            10.0 * np.exp(-0.1 * np.arange(20)),  # Decreasing
            [15.0],  # Sudden spike
            10.0 * np.exp(-0.1 * np.arange(20, 50))  # Continue
        ])

        result = self.analyzer.detect_divergence(losses)

        self.assertGreater(result['count'], 0)
        self.assertEqual(result['severity'], 'critical')
        self.assertEqual(result['points'][0]['step'], 20)
        self.assertGreater(result['points'][0]['increase_pct'], 15)

    def test_gradual_divergence(self):
        """Test detection of gradual loss increase."""
        # Decrease then gradual increase
        losses = np.concatenate([
            10.0 * np.exp(-0.1 * np.arange(25)),  # Decreasing
            5.0 + 0.5 * np.arange(25)  # Increasing
        ])

        result = self.analyzer.detect_divergence(losses)

        # Should detect multiple divergence points in the increasing section
        self.assertGreater(result['count'], 0)
        self.assertIn(result['severity'], ['critical', 'high'])

    def test_noisy_but_healthy(self):
        """Test that noisy but decreasing loss is not over-flagged."""
        # Add noise to decreasing trend
        np.random.seed(42)
        base_losses = 10.0 * np.exp(-0.1 * np.arange(50))
        noise = np.random.normal(0, 0.1, 50)
        losses = base_losses + noise

        result = self.analyzer.detect_divergence(losses)

        # Should have minimal or no divergence detections
        self.assertLess(result['count'], 3)

    def test_multiple_spikes(self):
        """Test detection of multiple divergence points."""
        # Create losses with multiple spikes
        losses = 5.0 * np.exp(-0.05 * np.arange(100))
        losses[20] *= 2.0  # First spike
        losses[50] *= 2.5  # Second spike
        losses[80] *= 1.8  # Third spike

        result = self.analyzer.detect_divergence(losses)

        # Should detect at least 2 spikes
        self.assertGreaterEqual(result['count'], 2)

        # Check that spikes are ordered
        spike_steps = [p['step'] for p in result['points']]
        self.assertEqual(spike_steps, sorted(spike_steps))

    def test_edge_case_short_sequence(self):
        """Test behavior with very short loss sequences."""
        losses = np.array([5.0, 4.5, 4.0])

        result = self.analyzer.detect_divergence(losses)

        # Should handle gracefully without errors
        self.assertIsInstance(result, dict)
        self.assertIn('count', result)
        self.assertIn('severity', result)

    def test_edge_case_constant_loss(self):
        """Test behavior with constant (plateaued) loss."""
        losses = np.ones(50) * 3.5

        result = self.analyzer.detect_divergence(losses)

        # Constant loss shouldn't trigger divergence
        self.assertEqual(result['count'], 0)

    def test_realistic_phase1_scenario(self):
        """Test realistic Phase 1 training divergence scenario."""
        # Simulate the actual Phase 1 failure:
        # Loss decreases for 6 epochs, then increases 15% at epoch 7
        epochs = np.arange(15)

        # Healthy decrease for first 6 epochs
        losses = 4.5 * np.exp(-0.15 * epochs[:7])

        # 15% increase at epoch 7
        divergence_loss = losses[6] * 1.15

        # Continue with unstable training
        post_divergence = divergence_loss + 0.05 * np.random.randn(8)

        losses = np.concatenate([losses, [divergence_loss], post_divergence])

        result = self.analyzer.detect_divergence(losses)

        # Should detect divergence around epoch 7
        self.assertGreater(result['count'], 0)
        self.assertIn(result['severity'], ['critical', 'high'])

        # First divergence should be near epoch 7
        first_divergence = result['points'][0]
        self.assertGreaterEqual(first_divergence['step'], 6)
        self.assertLessEqual(first_divergence['step'], 8)
        self.assertGreater(first_divergence['increase_pct'], 10)


class TestPlateauDetection(unittest.TestCase):
    """Test cases for plateau detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = LossAnalyzer()

    def test_no_plateau(self):
        """Test that continuously decreasing loss has no plateau."""
        losses = 10.0 * np.exp(-0.1 * np.arange(50))
        result = self.analyzer.detect_plateau(losses)

        self.assertEqual(result['count'], 0)

    def test_single_plateau(self):
        """Test detection of single plateau region."""
        # Decrease, plateau, decrease
        losses = np.concatenate([
            10.0 * np.exp(-0.2 * np.arange(15)),  # Decrease
            np.ones(20) * 2.23,  # Plateau
            2.23 * np.exp(-0.1 * np.arange(15))  # Decrease again
        ])

        result = self.analyzer.detect_plateau(losses, min_length=10)

        self.assertGreaterEqual(result['count'], 1)

        # Check plateau is in middle region
        plateau = result['regions'][0]
        self.assertGreaterEqual(plateau['start'], 10)
        self.assertLessEqual(plateau['end'], 40)
        self.assertGreaterEqual(plateau['length'], 10)

    def test_multiple_plateaus(self):
        """Test detection of multiple plateau regions."""
        losses = np.concatenate([
            10.0 * np.exp(-0.3 * np.arange(10)),  # Decrease
            np.ones(15) * 5.0,  # First plateau
            5.0 * np.exp(-0.2 * np.arange(10)),  # Decrease
            np.ones(15) * 2.0,  # Second plateau
            2.0 * np.exp(-0.1 * np.arange(10))  # Final decrease
        ])

        result = self.analyzer.detect_plateau(losses, min_length=10)

        self.assertGreaterEqual(result['count'], 2)


class TestNoiseAnalysis(unittest.TestCase):
    """Test cases for noise analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = LossAnalyzer()

    def test_low_noise(self):
        """Test analysis of low-noise loss curve."""
        np.random.seed(42)
        base_losses = 10.0 * np.exp(-0.1 * np.arange(50))
        noise = np.random.normal(0, 0.01, 50)
        losses = base_losses + noise

        result = self.analyzer.analyze_noise(losses)

        # Low noise should have high SNR
        self.assertGreater(result['snr'], 10)
        self.assertLess(result['std'], 0.5)

    def test_high_noise(self):
        """Test analysis of high-noise loss curve."""
        np.random.seed(42)
        base_losses = 5.0 * np.ones(50)
        noise = np.random.normal(0, 1.0, 50)
        losses = base_losses + noise

        result = self.analyzer.analyze_noise(losses)

        # High noise should have low SNR
        self.assertLess(result['snr'], 10)
        self.assertGreater(result['std'], 0.5)


def run_tests():
    """Run all test suites."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestLossDivergence))
    suite.addTests(loader.loadTestsFromTestCase(TestPlateauDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestNoiseAnalysis))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
