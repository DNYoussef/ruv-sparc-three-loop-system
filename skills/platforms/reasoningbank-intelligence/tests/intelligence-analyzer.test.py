#!/usr/bin/env python3
"""
Unit tests for Intelligence Analyzer

Tests learning curve analysis, strategy evolution tracking,
and meta-learning insights.
"""

import unittest
import json
import tempfile
import os
from datetime import datetime, timedelta
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'resources'))

from intelligence_analyzer import IntelligenceAnalyzer, LearningMetrics


class TestIntelligenceAnalyzer(unittest.TestCase):
    """Test suite for IntelligenceAnalyzer"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = IntelligenceAnalyzer(window_size=20)
        self.sample_experiences = self._create_sample_experiences()

    def _create_sample_experiences(self):
        """Create sample experiences with learning progression"""
        base_time = datetime.now().timestamp()
        experiences = []

        # Simulate improving performance over time
        for i in range(100):
            success_probability = 0.3 + (i / 100) * 0.5  # 30% -> 80%
            success = (i % 10) < (success_probability * 10)

            experiences.append({
                'task_id': f'task_{i}',
                'task_type': 'code_review',
                'approach': 'static_analysis' if i < 50 else 'tdd_approach',
                'outcome': {'success': success, 'bugs_found': 5 if success else 1},
                'context': {
                    'language': 'python' if i % 3 == 0 else 'javascript',
                    'complexity': 'medium'
                },
                'timestamp': base_time + i * 3600,
                'duration': 100 + (i % 20) * 5
            })

        return experiences

    def test_load_experiences(self):
        """Test loading experiences from file"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(self.sample_experiences[:10], f)
            temp_path = f.name

        self.analyzer.load_experiences(temp_path)
        self.assertEqual(len(self.analyzer.experiences), 10)

        os.unlink(temp_path)

    def test_learning_curve_analysis(self):
        """Test learning curve detection"""
        self.analyzer.experiences = self.sample_experiences

        curve = self.analyzer.analyze_learning_curve()

        # Should detect improving trend
        self.assertIn('trend', curve)
        self.assertIn('improvement_rate', curve)
        self.assertIn('r_squared', curve)

        # With improving data, should detect 'improving' trend
        self.assertEqual(curve['trend'], 'improving')
        self.assertGreater(curve['improvement_rate'], 0)

    def test_learning_curve_insufficient_data(self):
        """Test learning curve with insufficient data"""
        self.analyzer.experiences = self.sample_experiences[:5]

        curve = self.analyzer.analyze_learning_curve()
        self.assertIn('error', curve)

    def test_strategy_evolution(self):
        """Test strategy evolution tracking"""
        self.analyzer.experiences = self.sample_experiences

        evolution = self.analyzer.analyze_strategy_evolution()

        self.assertIn('evolution', evolution)
        self.assertIn('strategy_shifts', evolution)
        self.assertIn('num_strategies_tried', evolution)
        self.assertIn('dominant_strategy', evolution)

        # Should have multiple periods
        self.assertGreater(len(evolution['evolution']), 0)

    def test_detect_strategy_shifts(self):
        """Test detection of strategy shifts"""
        self.analyzer.experiences = self.sample_experiences

        evolution = self.analyzer.analyze_strategy_evolution()
        shifts = evolution['strategy_shifts']

        # With our test data (switching at i=50), should detect shift
        self.assertGreater(len(shifts), 0)

    def test_dominant_strategy(self):
        """Test finding dominant strategy"""
        self.analyzer.experiences = self.sample_experiences

        evolution = self.analyzer.analyze_strategy_evolution()
        dominant = evolution['dominant_strategy']

        self.assertIn('strategy', dominant)
        self.assertIn('usage_count', dominant)
        self.assertIn('success_rate', dominant)

    def test_performance_regressions(self):
        """Test detection of performance regressions"""
        # Create experiences with a regression
        experiences = []
        base_time = datetime.now().timestamp()

        # Good performance
        for i in range(50):
            experiences.append({
                'task_id': f'task_{i}',
                'task_type': 'test',
                'approach': 'approach_a',
                'outcome': {'success': i % 10 < 8},  # 80% success
                'context': {},
                'timestamp': base_time + i * 60,
                'duration': 60
            })

        # Performance drop
        for i in range(50, 100):
            experiences.append({
                'task_id': f'task_{i}',
                'task_type': 'test',
                'approach': 'approach_a',
                'outcome': {'success': i % 10 < 4},  # 40% success
                'context': {},
                'timestamp': base_time + i * 60,
                'duration': 60
            })

        self.analyzer.experiences = experiences

        regressions = self.analyzer.detect_performance_regressions()

        # Should detect regression
        self.assertGreater(len(regressions), 0)

        if regressions:
            regression = regressions[0]
            self.assertIn('previous_rate', regression)
            self.assertIn('current_rate', regression)
            self.assertIn('p_value', regression)
            self.assertLess(regression['p_value'], 0.05)

    def test_transfer_opportunities(self):
        """Test identification of transfer learning opportunities"""
        # Create experiences with successful strategy for one task type
        experiences = []
        base_time = datetime.now().timestamp()

        context_a = {'domain': 'web', 'language': 'javascript'}
        context_b = {'domain': 'web', 'language': 'typescript'}  # Similar context

        # Task A with lots of data and successful strategy
        for i in range(30):
            experiences.append({
                'task_id': f'task_a_{i}',
                'task_type': 'task_a',
                'approach': 'strategy_x',
                'outcome': {'success': i % 10 < 8},  # 80% success
                'context': context_a,
                'timestamp': base_time + i * 60,
                'duration': 60
            })

        # Task B with little data
        for i in range(3):
            experiences.append({
                'task_id': f'task_b_{i}',
                'task_type': 'task_b',
                'approach': 'strategy_y',
                'outcome': {'success': False},
                'context': context_b,
                'timestamp': base_time + (30 + i) * 60,
                'duration': 60
            })

        self.analyzer.experiences = experiences

        opportunities = self.analyzer.identify_transfer_opportunities()

        # Should identify transfer opportunity from task_a to task_b
        # (similar contexts, task_a has successful strategy, task_b has little data)
        self.assertIsInstance(opportunities, list)

    def test_learning_metrics(self):
        """Test calculation of learning metrics"""
        self.analyzer.experiences = self.sample_experiences

        metrics = self.analyzer.get_learning_metrics()

        self.assertIsInstance(metrics, LearningMetrics)
        self.assertEqual(metrics.total_experiences, len(self.sample_experiences))
        self.assertGreater(metrics.success_rate, 0)
        self.assertGreater(metrics.strategy_diversity, 0)
        self.assertGreaterEqual(metrics.confidence_level, 0)

    def test_export_report(self):
        """Test comprehensive report export"""
        self.analyzer.experiences = self.sample_experiences

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            report_path = f.name

        self.analyzer.export_report(report_path)

        # Verify report was created
        self.assertTrue(os.path.exists(report_path))

        # Load and verify content
        with open(report_path, 'r') as f:
            report = json.load(f)

        self.assertIn('timestamp', report)
        self.assertIn('metrics', report)
        self.assertIn('learning_curve', report)
        self.assertIn('strategy_evolution', report)
        self.assertIn('performance_regressions', report)
        self.assertIn('transfer_opportunities', report)

        os.unlink(report_path)

    def test_empty_experiences(self):
        """Test behavior with no experiences"""
        metrics = self.analyzer.get_learning_metrics()
        self.assertEqual(metrics.total_experiences, 0)
        self.assertEqual(metrics.success_rate, 0)

    def test_adaptation_speed(self):
        """Test calculation of adaptation speed"""
        # Create experiences with strategy changes
        experiences = []
        base_time = datetime.now().timestamp()

        strategies = ['a', 'b', 'c', 'a', 'b', 'c']  # Frequent changes
        for i, strategy in enumerate(strategies * 10):
            experiences.append({
                'task_id': f'task_{i}',
                'task_type': 'test',
                'approach': strategy,
                'outcome': {'success': True},
                'context': {},
                'timestamp': base_time + i * 60,
                'duration': 60
            })

        self.analyzer.experiences = experiences
        metrics = self.analyzer.get_learning_metrics()

        # Should have high adaptation speed (many strategy changes)
        self.assertGreater(metrics.adaptation_speed, 0)


class TestLearningMetrics(unittest.TestCase):
    """Test LearningMetrics dataclass"""

    def test_metrics_creation(self):
        """Test creation of LearningMetrics"""
        metrics = LearningMetrics(
            total_experiences=100,
            success_rate=0.75,
            improvement_rate=0.01,
            learning_efficiency=0.0075,
            strategy_diversity=0.6,
            adaptation_speed=0.15,
            confidence_level=0.95
        )

        self.assertEqual(metrics.total_experiences, 100)
        self.assertEqual(metrics.success_rate, 0.75)
        self.assertEqual(metrics.confidence_level, 0.95)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
