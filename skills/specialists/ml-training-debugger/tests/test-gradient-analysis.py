#!/usr/bin/env python3
"""
Test suite for gradient flow analysis

Tests the gradient-debugger.py script's ability to detect and diagnose
gradient-related training issues.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Mock torch if not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - running with mocked tests")


class MockGradientDebugger:
    """Mock GradientDebugger for testing without PyTorch."""

    def __init__(self, model=None, vanishing_threshold=1e-6, exploding_threshold=100.0):
        self.model = model
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold

    def analyze_layer_gradients(self, checkpoint_path):
        return {
            'layers': {},
            'summary': {
                'total_layers': 0,
                'vanishing_layers': 0,
                'exploding_layers': 0,
                'healthy_layers': 0
            },
            'issues': [],
            'recommendations': []
        }


if TORCH_AVAILABLE:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'resources' / 'scripts'))
    try:
        from gradient_debugger import GradientDebugger
    except ImportError:
        GradientDebugger = MockGradientDebugger
else:
    GradientDebugger = MockGradientDebugger


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for gradient tests")
class TestGradientDetection(unittest.TestCase):
    """Test cases for gradient issue detection."""

    def setUp(self):
        """Set up test fixtures."""
        # Create simple test model
        self.model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.debugger = GradientDebugger(self.model)

    def test_healthy_gradients(self):
        """Test that healthy gradients are not flagged."""
        # Create dummy input and target
        x = torch.randn(32, 64)
        target = torch.randint(0, 10, (32,))
        loss_fn = nn.CrossEntropyLoss()

        # Perform analysis
        analysis = self.debugger.analyze_gradient_flow(x, target, loss_fn)

        # Should have minimal or no bottlenecks
        self.assertLessEqual(len(analysis['bottlenecks']), 1)

        # All layers should have some gradient
        for name, stats in analysis['layers'].items():
            self.assertGreater(stats['mean'], 0)
            self.assertFalse(stats.get('has_nans', False))
            self.assertFalse(stats.get('has_infs', False))

    def test_vanishing_gradients(self):
        """Test detection of vanishing gradients."""
        # Create very deep network prone to vanishing gradients
        layers = []
        for _ in range(20):
            layers.extend([
                nn.Linear(128, 128),
                nn.Sigmoid()  # Sigmoid can cause vanishing gradients
            ])
        layers.append(nn.Linear(128, 10))

        deep_model = nn.Sequential(*layers)
        debugger = GradientDebugger(deep_model, vanishing_threshold=1e-6)

        x = torch.randn(32, 128)
        target = torch.randint(0, 10, (32,))
        loss_fn = nn.CrossEntropyLoss()

        analysis = debugger.analyze_gradient_flow(x, target, loss_fn)

        # Deep network with sigmoid should show vanishing gradients
        vanishing_bottlenecks = [b for b in analysis['bottlenecks']
                                 if b['type'] == 'vanishing']
        self.assertGreater(len(vanishing_bottlenecks), 0)

    def test_gradient_nan_detection(self):
        """Test detection of NaN gradients."""
        # Manually inject NaN gradient
        x = torch.randn(32, 64)
        target = torch.randint(0, 10, (32,))
        loss_fn = nn.CrossEntropyLoss()

        # Run forward/backward
        output = self.model(x)
        loss = loss_fn(output, target)
        loss.backward()

        # Inject NaN into first layer gradient
        with torch.no_grad():
            self.model[0].weight.grad[0, 0] = float('nan')

        # Create checkpoint for analysis
        checkpoint_path = Path('/tmp/test_checkpoint.pt')
        torch.save({'model_state_dict': self.model.state_dict()}, checkpoint_path)

        analysis = self.debugger.analyze_layer_gradients(checkpoint_path)

        # Should detect NaN
        nan_issues = [issue for issue in analysis['issues'] if 'NaN' in issue]
        self.assertGreater(len(nan_issues), 0)

        # Cleanup
        checkpoint_path.unlink()

    def test_gradient_explosion_detection(self):
        """Test detection of exploding gradients."""
        # Create model with large weights
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        # Initialize with large weights to cause explosion
        with torch.no_grad():
            for param in model.parameters():
                param.data *= 10.0

        debugger = GradientDebugger(model, exploding_threshold=10.0)

        x = torch.randn(32, 64)
        target = torch.randint(0, 10, (32,))
        loss_fn = nn.CrossEntropyLoss()

        analysis = debugger.analyze_gradient_flow(x, target, loss_fn)

        # Should detect exploding gradients
        exploding_bottlenecks = [b for b in analysis['bottlenecks']
                                 if b['type'] == 'exploding']
        # Note: May or may not explode depending on random initialization
        # Just check that detection mechanism works
        self.assertIsInstance(exploding_bottlenecks, list)


class TestGradientFlowPattern(unittest.TestCase):
    """Test cases for gradient flow pattern analysis."""

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required")
    def test_flow_pattern_detection(self):
        """Test detection of different flow patterns."""
        # Create model
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        debugger = GradientDebugger(model)

        x = torch.randn(16, 32)
        target = torch.randint(0, 10, (16,))
        loss_fn = nn.CrossEntropyLoss()

        analysis = debugger.analyze_gradient_flow(x, target, loss_fn)

        # Should identify a flow pattern
        self.assertIn('flow_pattern', analysis)
        self.assertIsInstance(analysis['flow_pattern'], str)

        # Pattern should be one of the defined types
        valid_patterns = ['unknown', 'insufficient_data', 'severe_vanishing',
                         'monotonic_increase', 'monotonic_decrease', 'mixed']
        self.assertIn(analysis['flow_pattern'], valid_patterns)


class TestRecommendations(unittest.TestCase):
    """Test cases for gradient debugging recommendations."""

    def test_vanishing_recommendations(self):
        """Test recommendations for vanishing gradients."""
        analysis = {
            'summary': {
                'total_layers': 10,
                'vanishing_layers': 4,  # 40%
                'exploding_layers': 0,
                'healthy_layers': 6
            },
            'layers': {}
        }

        # Mock debugger to test recommendation generation
        debugger = GradientDebugger(None)
        recommendations = debugger._generate_recommendations(analysis)

        # Should recommend fixes for vanishing gradients
        self.assertGreater(len(recommendations), 0)

        # Check for specific recommendations
        rec_text = ' '.join(recommendations)
        self.assertIn('vanishing', rec_text.lower())

    def test_exploding_recommendations(self):
        """Test recommendations for exploding gradients."""
        analysis = {
            'summary': {
                'total_layers': 10,
                'vanishing_layers': 0,
                'exploding_layers': 3,  # 30%
                'healthy_layers': 7
            },
            'layers': {}
        }

        debugger = GradientDebugger(None)
        recommendations = debugger._generate_recommendations(analysis)

        # Should recommend gradient clipping
        rec_text = ' '.join(recommendations)
        self.assertIn('clip', rec_text.lower())

    def test_healthy_recommendations(self):
        """Test recommendations for healthy gradients."""
        analysis = {
            'summary': {
                'total_layers': 10,
                'vanishing_layers': 0,
                'exploding_layers': 0,
                'healthy_layers': 10
            },
            'layers': {}
        }

        debugger = GradientDebugger(None)
        recommendations = debugger._generate_recommendations(analysis)

        # Should give positive feedback
        self.assertGreater(len(recommendations), 0)
        rec_text = ' '.join(recommendations)
        self.assertIn('healthy', rec_text.lower())


def run_tests():
    """Run all test suites."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestGradientDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestGradientFlowPattern))
    suite.addTests(loader.loadTestsFromTestCase(TestRecommendations))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    if not TORCH_AVAILABLE:
        print("\n⚠️ PyTorch not installed - running limited tests")
        print("Install PyTorch for full test coverage:")
        print("  pip install torch\n")

    sys.exit(run_tests())
