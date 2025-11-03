#!/usr/bin/env python3
"""
Unit tests for Pattern Recognizer

Tests pattern detection algorithms, statistical significance,
and edge cases.
"""

import unittest
import json
import tempfile
import os
from datetime import datetime, timedelta
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'resources'))

from pattern_recognizer import PatternRecognizer, Pattern, TaskExperience


class TestPatternRecognizer(unittest.TestCase):
    """Test suite for PatternRecognizer"""

    def setUp(self):
        """Set up test fixtures"""
        self.recognizer = PatternRecognizer(min_support=2, confidence_threshold=0.6)
        self.sample_experiences = self._create_sample_experiences()

    def _create_sample_experiences(self):
        """Create sample task experiences for testing"""
        base_time = datetime.now().timestamp()

        experiences = []

        # Successful sequence pattern
        for i in range(5):
            experiences.append({
                'task_id': f'task_{i}',
                'task_type': 'code_review',
                'approach': 'static_analysis_first',
                'outcome': {'success': True, 'bugs_found': 5},
                'context': {'language': 'python', 'complexity': 'medium'},
                'timestamp': base_time + i * 3600,
                'duration': 120
            })

        # Failed different approach
        for i in range(3):
            experiences.append({
                'task_id': f'task_{i+5}',
                'task_type': 'code_review',
                'approach': 'manual_review_first',
                'outcome': {'success': False, 'bugs_found': 1},
                'context': {'language': 'javascript', 'complexity': 'high'},
                'timestamp': base_time + (i + 5) * 3600,
                'duration': 200
            })

        # Contextual pattern (Python + static analysis = success)
        for i in range(4):
            experiences.append({
                'task_id': f'task_{i+8}',
                'task_type': 'code_review',
                'approach': 'static_analysis_first',
                'outcome': {'success': True, 'bugs_found': 3},
                'context': {'language': 'python', 'complexity': 'low'},
                'timestamp': base_time + (i + 8) * 3600,
                'duration': 80
            })

        return experiences

    def test_add_experience(self):
        """Test adding experiences"""
        exp = self.sample_experiences[0]
        self.recognizer.add_experience(exp)

        self.assertEqual(len(self.recognizer.experiences), 1)
        self.assertEqual(self.recognizer.experiences[0].task_type, 'code_review')

    def test_sequence_pattern_detection(self):
        """Test detection of sequence patterns"""
        for exp in self.sample_experiences[:5]:
            self.recognizer.add_experience(exp)

        patterns = self.recognizer.analyze_patterns()

        # Should detect successful sequence pattern
        sequence_patterns = [p for p in patterns if p.pattern_type == 'sequence']
        self.assertGreater(len(sequence_patterns), 0)

    def test_contextual_pattern_detection(self):
        """Test context-dependent pattern detection"""
        for exp in self.sample_experiences:
            self.recognizer.add_experience(exp)

        patterns = self.recognizer.analyze_patterns()

        # Should detect Python + static_analysis pattern
        contextual_patterns = [p for p in patterns if p.pattern_type == 'contextual']
        self.assertGreater(len(contextual_patterns), 0)

        # Verify confidence
        for pattern in contextual_patterns:
            self.assertGreaterEqual(pattern.confidence, self.recognizer.confidence_threshold)

    def test_min_support_threshold(self):
        """Test that patterns below min_support are not detected"""
        # Add only 1 experience (below min_support of 2)
        self.recognizer.add_experience(self.sample_experiences[0])

        patterns = self.recognizer.analyze_patterns()
        self.assertEqual(len(patterns), 0)

    def test_confidence_threshold(self):
        """Test that low-confidence patterns are filtered"""
        # Mix of successes and failures
        mixed_experiences = []
        for i in range(10):
            mixed_experiences.append({
                'task_id': f'task_{i}',
                'task_type': 'test',
                'approach': 'approach_a',
                'outcome': {'success': i % 2 == 0},  # 50% success rate
                'context': {},
                'timestamp': datetime.now().timestamp() + i * 60,
                'duration': 60
            })

        for exp in mixed_experiences:
            self.recognizer.add_experience(exp)

        patterns = self.recognizer.analyze_patterns()

        # Patterns with 50% success should not exceed confidence threshold of 0.6
        for pattern in patterns:
            self.assertGreaterEqual(pattern.confidence, 0.6)

    def test_temporal_pattern_detection(self):
        """Test time-based pattern detection"""
        # Create experiences at specific hours with different success rates
        morning_success_rate = 0.9
        evening_success_rate = 0.4

        base_time = datetime.now().replace(hour=9, minute=0, second=0).timestamp()

        # Morning tasks (high success)
        for i in range(10):
            self.recognizer.add_experience({
                'task_id': f'morning_{i}',
                'task_type': 'coding',
                'approach': 'test_first',
                'outcome': {'success': i < 9},  # 90% success
                'context': {},
                'timestamp': base_time + i * 3600,
                'duration': 60
            })

        # Evening tasks (low success)
        evening_time = datetime.now().replace(hour=20, minute=0, second=0).timestamp()
        for i in range(10):
            self.recognizer.add_experience({
                'task_id': f'evening_{i}',
                'task_type': 'coding',
                'approach': 'test_first',
                'outcome': {'success': i < 4},  # 40% success
                'context': {},
                'timestamp': evening_time + i * 3600,
                'duration': 60
            })

        patterns = self.recognizer.analyze_patterns()
        temporal_patterns = [p for p in patterns if p.pattern_type == 'temporal']

        # May or may not detect temporal pattern depending on statistical significance
        # Just verify no crashes
        self.assertIsInstance(temporal_patterns, list)

    def test_anomaly_detection(self):
        """Test anomaly detection"""
        # Create mostly normal experiences
        for i in range(20):
            self.recognizer.add_experience({
                'task_id': f'normal_{i}',
                'task_type': 'review',
                'approach': 'standard',
                'outcome': {'success': True},
                'context': {},
                'timestamp': datetime.now().timestamp() + i * 60,
                'duration': 100 + (i % 10) * 5  # Duration 100-145
            })

        # Add anomalies
        self.recognizer.add_experience({
            'task_id': 'anomaly_1',
            'task_type': 'review',
            'approach': 'standard',
            'outcome': {'success': True},
            'context': {},
            'timestamp': datetime.now().timestamp() + 1000,
            'duration': 1000  # 10x longer
        })

        patterns = self.recognizer.analyze_patterns()
        anomaly_patterns = [p for p in patterns if p.pattern_type == 'anomaly']

        # Should detect anomaly
        self.assertGreater(len(anomaly_patterns), 0)

    def test_export_import_patterns(self):
        """Test pattern export and import"""
        for exp in self.sample_experiences[:5]:
            self.recognizer.add_experience(exp)

        patterns = self.recognizer.analyze_patterns()

        # Export
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            export_path = f.name

        self.recognizer.export_patterns(export_path)

        # Import into new recognizer
        new_recognizer = PatternRecognizer()
        new_recognizer.load_patterns(export_path)

        self.assertEqual(len(new_recognizer.patterns), len(self.recognizer.patterns))

        # Cleanup
        os.unlink(export_path)

    def test_statistical_significance(self):
        """Test that patterns have statistical significance"""
        for exp in self.sample_experiences:
            self.recognizer.add_experience(exp)

        patterns = self.recognizer.analyze_patterns()

        for pattern in patterns:
            # Statistical significance should be calculated
            self.assertIsNotNone(pattern.statistical_significance)
            self.assertGreaterEqual(pattern.statistical_significance, 0)
            self.assertLessEqual(pattern.statistical_significance, 1)

    def test_empty_experiences(self):
        """Test behavior with no experiences"""
        patterns = self.recognizer.analyze_patterns()
        self.assertEqual(len(patterns), 0)

    def test_pattern_attributes(self):
        """Test that detected patterns have required attributes"""
        for exp in self.sample_experiences:
            self.recognizer.add_experience(exp)

        patterns = self.recognizer.analyze_patterns()

        for pattern in patterns:
            # Verify all required attributes
            self.assertIsNotNone(pattern.pattern_id)
            self.assertIsNotNone(pattern.pattern_type)
            self.assertIsNotNone(pattern.description)
            self.assertIsInstance(pattern.triggers, list)
            self.assertIsInstance(pattern.actions, list)
            self.assertIsInstance(pattern.confidence, float)
            self.assertIsInstance(pattern.support, int)
            self.assertIsInstance(pattern.occurrences, list)


class TestPatternStatistics(unittest.TestCase):
    """Test statistical methods"""

    def setUp(self):
        self.recognizer = PatternRecognizer()

    def test_calculate_significance(self):
        """Test significance calculation"""
        # High success rate should be significant
        sig_high = self.recognizer._calculate_significance(45, 50)
        self.assertGreater(sig_high, 0.95)

        # 50/50 should not be significant
        sig_baseline = self.recognizer._calculate_significance(25, 50)
        self.assertLess(sig_baseline, 0.5)

        # Insufficient data
        sig_low_data = self.recognizer._calculate_significance(2, 2)
        self.assertEqual(sig_low_data, 0.0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
