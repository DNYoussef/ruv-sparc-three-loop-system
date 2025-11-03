#!/usr/bin/env python3
"""
Test Suite for Swarm Performance Analyzer
Comprehensive tests for metrics collection and analysis
"""

import unittest
import sys
import json
from datetime import datetime, timedelta
sys.path.insert(0, '../resources')

from swarm_analyzer import SwarmAnalyzer, AgentMetrics, SwarmMetrics


class TestSwarmAnalyzer(unittest.TestCase):
    """Test cases for SwarmAnalyzer"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = SwarmAnalyzer('test-swarm-001', time_window=3600)

    def test_initialization(self):
        """Test analyzer initialization"""
        self.assertEqual(self.analyzer.swarm_id, 'test-swarm-001')
        self.assertEqual(self.analyzer.time_window, 3600)
        self.assertEqual(len(self.analyzer.metrics_buffer), 0)
        self.assertEqual(len(self.analyzer.agent_metrics), 0)

    def test_collect_metrics(self):
        """Test metrics collection"""
        test_data = {
            'agent_id': 'agent-1',
            'type': 'coder',
            'task_time': 45.5,
            'cpu_usage': 0.65,
            'memory_usage': 0.50,
            'message_count': 10,
            'response_time': 1.5,
            'completed': 1,
            'failed': 0
        }

        self.analyzer.collect_metrics(test_data)

        self.assertEqual(len(self.analyzer.metrics_buffer), 1)
        self.assertIn('agent-1', self.analyzer.agent_metrics)
        self.assertEqual(len(self.analyzer.agent_metrics['agent-1']), 1)

    def test_analyze_agent_performance(self):
        """Test agent performance analysis"""
        # Collect sample metrics
        for i in range(5):
            self.analyzer.collect_metrics({
                'agent_id': 'agent-1',
                'type': 'coder',
                'task_time': 40 + i * 5,
                'cpu_usage': 0.6 + i * 0.05,
                'memory_usage': 0.5 + i * 0.05,
                'message_count': 8 + i,
                'response_time': 1.0 + i * 0.2,
                'completed': 1,
                'failed': 0
            })

        metrics = self.analyzer.analyze_agent_performance('agent-1')

        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, AgentMetrics)
        self.assertEqual(metrics.agent_id, 'agent-1')
        self.assertEqual(metrics.agent_type, 'coder')
        self.assertEqual(metrics.tasks_completed, 5)
        self.assertEqual(metrics.tasks_failed, 0)
        self.assertGreater(metrics.avg_task_time, 0)

    def test_analyze_swarm_performance(self):
        """Test swarm-level performance analysis"""
        # Simulate multiple agents
        for agent_num in range(3):
            for i in range(5):
                self.analyzer.collect_metrics({
                    'agent_id': f'agent-{agent_num}',
                    'type': 'coder',
                    'task_time': 40 + i * 5,
                    'cpu_usage': 0.6,
                    'memory_usage': 0.5,
                    'message_count': 10,
                    'response_time': 1.5,
                    'completed': 1 if i % 4 != 0 else 0,
                    'failed': 1 if i % 4 == 0 else 0
                })

        swarm_metrics = self.analyzer.analyze_swarm_performance('mesh')

        self.assertIsInstance(swarm_metrics, SwarmMetrics)
        self.assertEqual(swarm_metrics.swarm_id, 'test-swarm-001')
        self.assertEqual(swarm_metrics.topology, 'mesh')
        self.assertEqual(swarm_metrics.agent_count, 3)
        self.assertGreater(swarm_metrics.total_tasks, 0)
        self.assertGreater(swarm_metrics.throughput, 0)

    def test_detect_bottlenecks(self):
        """Test bottleneck detection"""
        # Create conditions for bottlenecks
        self.analyzer.collect_metrics({
            'agent_id': 'slow-agent',
            'type': 'coder',
            'task_time': 75000,  # 75 seconds - slow
            'cpu_usage': 0.9,
            'memory_usage': 0.85,  # High memory
            'message_count': 100,
            'response_time': 3.5,  # Slow response
            'completed': 1,
            'failed': 0
        })

        self.analyzer.collect_metrics({
            'agent_id': 'underutilized-agent',
            'type': 'reviewer',
            'task_time': 30000,
            'cpu_usage': 0.3,
            'memory_usage': 0.4,
            'message_count': 5,
            'response_time': 0.8,
            'completed': 0,
            'failed': 0
        })

        bottlenecks = self.analyzer.detect_bottlenecks(threshold=0.2)

        self.assertIn('communication', bottlenecks)
        self.assertIn('processing', bottlenecks)
        self.assertIn('memory', bottlenecks)
        self.assertGreater(len(bottlenecks['communication']), 0)
        self.assertGreater(len(bottlenecks['processing']), 0)
        self.assertGreater(len(bottlenecks['memory']), 0)

    def test_generate_report_json(self):
        """Test JSON report generation"""
        # Add sample data
        self.analyzer.collect_metrics({
            'agent_id': 'agent-1',
            'type': 'coder',
            'task_time': 45,
            'cpu_usage': 0.6,
            'memory_usage': 0.5,
            'message_count': 10,
            'response_time': 1.5,
            'completed': 1,
            'failed': 0
        })

        report = self.analyzer.generate_report('mesh', format='json')

        self.assertIsInstance(report, str)
        # Validate JSON structure
        report_data = json.loads(report)
        self.assertIn('swarm_metrics', report_data)
        self.assertIn('bottlenecks', report_data)

    def test_generate_report_text(self):
        """Test text report generation"""
        # Add sample data
        for i in range(3):
            self.analyzer.collect_metrics({
                'agent_id': f'agent-{i}',
                'type': 'coder',
                'task_time': 45 + i * 5,
                'cpu_usage': 0.6,
                'memory_usage': 0.5,
                'message_count': 10,
                'response_time': 1.5,
                'completed': 1,
                'failed': 0
            })

        report = self.analyzer.generate_report('mesh', format='text')

        self.assertIsInstance(report, str)
        self.assertIn('SWARM PERFORMANCE ANALYSIS REPORT', report)
        self.assertIn('SUMMARY METRICS', report)
        self.assertIn('BOTTLENECK ANALYSIS', report)

    def test_metrics_cleanup(self):
        """Test old metrics cleanup"""
        # Add old metrics (simulate by manipulating buffer directly)
        old_time = datetime.now() - timedelta(hours=2)
        self.analyzer.metrics_buffer.append({
            'timestamp': old_time.isoformat(),
            'data': {'agent_id': 'old-agent'}
        })

        # Add recent metrics
        self.analyzer.collect_metrics({
            'agent_id': 'new-agent',
            'type': 'coder',
            'task_time': 45,
            'cpu_usage': 0.6,
            'memory_usage': 0.5,
            'message_count': 10,
            'response_time': 1.5,
            'completed': 1,
            'failed': 0
        })

        # Cleanup should remove old metrics
        self.assertEqual(len(self.analyzer.metrics_buffer), 1)
        # Verify recent metrics remain
        self.assertIn('new-agent', self.analyzer.agent_metrics)

    def test_empty_metrics_handling(self):
        """Test handling of empty metrics"""
        metrics = self.analyzer.analyze_agent_performance('nonexistent-agent')
        self.assertIsNone(metrics)

        swarm_metrics = self.analyzer.analyze_swarm_performance('mesh')
        self.assertIsInstance(swarm_metrics, SwarmMetrics)
        self.assertEqual(swarm_metrics.agent_count, 0)

    def test_high_load_scenario(self):
        """Test analyzer performance under high load"""
        # Simulate high metric volume
        for agent_id in range(10):
            for sample in range(100):
                self.analyzer.collect_metrics({
                    'agent_id': f'agent-{agent_id}',
                    'type': 'coder',
                    'task_time': 40 + sample % 20,
                    'cpu_usage': 0.5 + (sample % 40) / 100,
                    'memory_usage': 0.4 + (sample % 50) / 100,
                    'message_count': 8 + sample % 15,
                    'response_time': 1.0 + (sample % 10) / 10,
                    'completed': 1 if sample % 5 != 0 else 0,
                    'failed': 1 if sample % 5 == 0 else 0
                })

        # Verify analyzer can handle the load
        swarm_metrics = self.analyzer.analyze_swarm_performance('mesh')
        self.assertEqual(swarm_metrics.agent_count, 10)
        self.assertGreater(swarm_metrics.total_tasks, 0)

        bottlenecks = self.analyzer.detect_bottlenecks()
        self.assertIsInstance(bottlenecks, dict)


class TestAgentMetrics(unittest.TestCase):
    """Test cases for AgentMetrics dataclass"""

    def test_agent_metrics_creation(self):
        """Test AgentMetrics instantiation"""
        metrics = AgentMetrics(
            agent_id='test-agent',
            agent_type='coder',
            tasks_completed=10,
            tasks_failed=2,
            avg_task_time=45.5,
            cpu_usage=0.65,
            memory_usage=0.50,
            message_count=25,
            response_time=1.5,
            utilization=0.75
        )

        self.assertEqual(metrics.agent_id, 'test-agent')
        self.assertEqual(metrics.tasks_completed, 10)
        self.assertEqual(metrics.utilization, 0.75)


class TestSwarmMetrics(unittest.TestCase):
    """Test cases for SwarmMetrics dataclass"""

    def test_swarm_metrics_creation(self):
        """Test SwarmMetrics instantiation"""
        metrics = SwarmMetrics(
            swarm_id='test-swarm',
            topology='mesh',
            agent_count=5,
            total_tasks=50,
            completed_tasks=45,
            failed_tasks=5,
            avg_completion_time=42.5,
            throughput=15.5,
            efficiency_score=87.5,
            bottleneck_agents=['agent-1', 'agent-3'],
            timestamp=datetime.now().isoformat()
        )

        self.assertEqual(metrics.swarm_id, 'test-swarm')
        self.assertEqual(metrics.agent_count, 5)
        self.assertEqual(metrics.efficiency_score, 87.5)
        self.assertEqual(len(metrics.bottleneck_agents), 2)


def run_tests():
    """Run all test suites"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSwarmAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestSwarmMetrics))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
