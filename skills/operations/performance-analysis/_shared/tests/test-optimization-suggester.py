#!/usr/bin/env python3
"""
Tests for Optimization Suggester
"""

import unittest
import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "resources"))

from optimization_suggester import (
    OptimizationSuggester,
    Optimization,
    Severity,
    Category
)


class TestOptimizationSuggester(unittest.TestCase):
    """Test suite for OptimizationSuggester"""

    def setUp(self):
        """Set up test fixtures"""
        self.suggester = OptimizationSuggester()

    def test_initialization(self):
        """Test suggester initialization"""
        self.assertIsInstance(self.suggester, OptimizationSuggester)
        self.assertEqual(len(self.suggester.optimizations), 0)

    def test_cpu_analysis(self):
        """Test CPU profile analysis"""
        cpu_data = {
            'duration': 100,
            'top_functions': [
                {
                    'function': 'slow_function',
                    'cumulative_time': 80,
                    'calls': 1000
                },
                {
                    'function': 'fast_function',
                    'cumulative_time': 5,
                    'calls': 50000
                }
            ]
        }

        suggestions = self.suggester.analyze_cpu_profile(cpu_data)

        self.assertGreater(len(suggestions), 0)

        # Should detect hot path
        hot_path = [s for s in suggestions if 'Hot path' in s.title]
        self.assertGreater(len(hot_path), 0)
        self.assertEqual(hot_path[0].category, Category.CPU)

        # Should detect excessive calls
        excessive_calls = [s for s in suggestions if 'Excessive calls' in s.title]
        self.assertGreater(len(excessive_calls), 0)

    def test_memory_analysis(self):
        """Test memory profile analysis"""
        memory_data = {
            'delta_mb': 150,
            'hotspots': [
                {
                    'file': 'memory_hog.py:42',
                    'size_diff_mb': 50
                },
                {
                    'file': 'allocator.py:15',
                    'size_diff_mb': 30
                }
            ]
        }

        suggestions = self.suggester.analyze_memory_profile(memory_data)

        self.assertGreater(len(suggestions), 0)

        # Should detect memory growth
        memory_growth = [s for s in suggestions if 'memory growth' in s.title.lower()]
        self.assertGreater(len(memory_growth), 0)
        self.assertEqual(memory_growth[0].severity, Severity.HIGH)

        # Should detect hotspots
        hotspots = [s for s in suggestions if 'hotspot' in s.title.lower()]
        self.assertGreater(len(hotspots), 0)

    def test_io_analysis(self):
        """Test I/O profile analysis"""
        io_data = {
            'disk': {
                'read_mb': 2000,
                'write_mb': 500
            },
            'network': {
                'sent_mb': 300,
                'recv_mb': 400
            }
        }

        suggestions = self.suggester.analyze_io_profile(io_data)

        self.assertGreater(len(suggestions), 0)

        # Should detect high disk I/O
        disk_io = [s for s in suggestions if 'disk' in s.title.lower()]
        self.assertGreater(len(disk_io), 0)
        self.assertEqual(disk_io[0].category, Category.IO)

    def test_bottleneck_analysis(self):
        """Test bottleneck detection analysis"""
        bottleneck_data = {
            'bottlenecks': [
                {
                    'type': 'event_loop_lag',
                    'lag': 250,
                    'severity': 'high'
                },
                {
                    'type': 'slow_query',
                    'duration': 500,
                    'severity': 'high'
                }
            ]
        }

        suggestions = self.suggester.analyze_bottlenecks(bottleneck_data)

        self.assertGreater(len(suggestions), 0)

        # Should detect event loop issues
        event_loop = [s for s in suggestions if 'event loop' in s.title.lower()]
        self.assertGreater(len(event_loop), 0)
        self.assertEqual(event_loop[0].category, Category.PARALLELIZATION)

        # Should detect database issues
        db_issues = [s for s in suggestions if 'query' in s.title.lower()]
        self.assertGreater(len(db_issues), 0)
        self.assertEqual(db_issues[0].category, Category.DATABASE)

    def test_comprehensive_analysis(self):
        """Test comprehensive profile analysis"""
        profile_data = {
            'cpu': {
                'duration': 100,
                'top_functions': [
                    {
                        'function': 'compute_heavy',
                        'cumulative_time': 70,
                        'calls': 1000
                    }
                ]
            },
            'memory': {
                'delta_mb': 200,
                'hotspots': [
                    {
                        'file': 'allocator.py',
                        'size_diff_mb': 100
                    }
                ]
            },
            'io': {
                'disk': {'read_mb': 1500},
                'network': {'sent_mb': 600}
            }
        }

        suggestions = self.suggester.generate_suggestions(profile_data)

        self.assertGreater(len(suggestions), 0)

        # Should have suggestions from all categories
        categories = set(s.category for s in suggestions)
        self.assertIn(Category.CPU, categories)
        self.assertIn(Category.MEMORY, categories)

    def test_severity_sorting(self):
        """Test suggestions are sorted by severity"""
        profile_data = {
            'cpu': {
                'duration': 100,
                'top_functions': [
                    {
                        'function': 'hot_function',
                        'cumulative_time': 50,
                        'calls': 100
                    }
                ]
            },
            'memory': {
                'delta_mb': 200,
                'hotspots': []
            }
        }

        suggestions = self.suggester.generate_suggestions(profile_data)

        # Check sorting (high severity should come first)
        for i in range(len(suggestions) - 1):
            severity_order = {
                Severity.CRITICAL: 0,
                Severity.HIGH: 1,
                Severity.MEDIUM: 2,
                Severity.LOW: 3
            }
            current_priority = severity_order[suggestions[i].severity]
            next_priority = severity_order[suggestions[i + 1].severity]
            self.assertLessEqual(current_priority, next_priority)

    def test_report_generation(self):
        """Test optimization report generation"""
        # Add some test suggestions
        self.suggester.optimizations = [
            Optimization(
                category=Category.CPU,
                severity=Severity.HIGH,
                title="Test optimization",
                description="Test description",
                impact="High impact",
                effort="Medium effort"
            ),
            Optimization(
                category=Category.MEMORY,
                severity=Severity.MEDIUM,
                title="Memory optimization",
                description="Reduce memory usage",
                impact="Medium impact",
                effort="Low effort"
            )
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            report = self.suggester.generate_report(f.name)

            self.assertIn('total_suggestions', report)
            self.assertEqual(report['total_suggestions'], 2)
            self.assertEqual(report['by_severity']['high'], 1)
            self.assertEqual(report['by_severity']['medium'], 1)

            # Verify file was created
            with open(f.name) as rf:
                saved_report = json.load(rf)
                self.assertEqual(saved_report['total_suggestions'], 2)


class TestOptimizationClass(unittest.TestCase):
    """Test Optimization dataclass"""

    def test_optimization_creation(self):
        """Test creating an Optimization"""
        opt = Optimization(
            category=Category.CPU,
            severity=Severity.HIGH,
            title="Test",
            description="Test description",
            impact="High",
            effort="Medium"
        )

        self.assertEqual(opt.category, Category.CPU)
        self.assertEqual(opt.severity, Severity.HIGH)
        self.assertEqual(opt.title, "Test")

    def test_optimization_to_dict(self):
        """Test converting Optimization to dict"""
        opt = Optimization(
            category=Category.MEMORY,
            severity=Severity.CRITICAL,
            title="Critical issue",
            description="Fix this now",
            impact="Very high",
            effort="High",
            estimated_improvement="50%"
        )

        opt_dict = opt.to_dict()

        self.assertEqual(opt_dict['category'], 'memory')
        self.assertEqual(opt_dict['severity'], 'critical')
        self.assertEqual(opt_dict['title'], 'Critical issue')
        self.assertEqual(opt_dict['estimated_improvement'], '50%')


def run_optimization_suggester_tests():
    """Run all optimization suggester tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestOptimizationSuggester))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizationClass))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_optimization_suggester_tests()
    sys.exit(0 if success else 1)
