#!/usr/bin/env python3
"""
Tests for Performance Profiler
"""

import unittest
import sys
import os
import time
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "resources"))

from profiler import PerformanceProfiler


class TestPerformanceProfiler(unittest.TestCase):
    """Test suite for PerformanceProfiler"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.profiler = PerformanceProfiler(output_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cpu_profiling(self):
        """Test CPU profiling functionality"""
        def cpu_intensive_function():
            total = 0
            for i in range(1000000):
                total += i * i
            return total

        metrics = self.profiler.profile_cpu(cpu_intensive_function)

        self.assertIn('duration', metrics)
        self.assertIn('top_functions', metrics)
        self.assertIn('total_calls', metrics)
        self.assertGreater(metrics['duration'], 0)
        self.assertGreater(len(metrics['top_functions']), 0)

    def test_memory_profiling(self):
        """Test memory profiling functionality"""
        def memory_intensive_function():
            # Allocate some memory
            data = []
            for i in range(100000):
                data.append([i] * 100)
            return data

        metrics = self.profiler.profile_memory(memory_intensive_function)

        self.assertIn('start_mb', metrics)
        self.assertIn('end_mb', metrics)
        self.assertIn('delta_mb', metrics)
        self.assertIn('hotspots', metrics)
        self.assertGreater(metrics['delta_mb'], 0)

    def test_io_profiling(self):
        """Test I/O profiling functionality"""
        metrics = self.profiler.profile_io(duration=2)

        self.assertIn('process', metrics)
        self.assertIn('disk', metrics)
        self.assertIn('network', metrics)
        self.assertIn('read_bytes', metrics['process'])
        self.assertIn('write_bytes', metrics['process'])

    def test_system_monitoring(self):
        """Test system resource monitoring"""
        metrics = self.profiler.monitor_system(interval=0.1, samples=5)

        self.assertIn('cpu', metrics)
        self.assertIn('memory', metrics)
        self.assertIn('disk', metrics)
        self.assertEqual(len(metrics['cpu']['samples']), 5)
        self.assertGreater(metrics['cpu']['avg'], 0)

    def test_process_profiling(self):
        """Test process profiling"""
        processes = self.profiler.profile_processes()

        self.assertIsInstance(processes, list)
        self.assertGreater(len(processes), 0)

        for proc in processes[:5]:
            self.assertIn('pid', proc)
            self.assertIn('name', proc)
            self.assertIn('cpu_percent', proc)
            self.assertIn('memory_percent', proc)

    def test_report_generation(self):
        """Test report generation"""
        # Generate some metrics
        self.profiler.profile_processes()

        report_file = self.profiler.generate_report()

        self.assertTrue(os.path.exists(report_file))
        self.assertTrue(report_file.endswith('.json'))

    def test_profile_context(self):
        """Test profiling context manager"""
        with self.profiler.profile_context(mode="all") as p:
            # Do some work
            data = [i * i for i in range(100000)]
            time.sleep(0.1)

        # Context manager should not raise exceptions
        self.assertIsInstance(p, PerformanceProfiler)

    def test_multiple_profiling_runs(self):
        """Test multiple profiling runs don't interfere"""
        def test_function():
            return sum(range(10000))

        metrics1 = self.profiler.profile_cpu(test_function)
        metrics2 = self.profiler.profile_cpu(test_function)

        self.assertIsInstance(metrics1, dict)
        self.assertIsInstance(metrics2, dict)
        # Both should have similar structure
        self.assertEqual(set(metrics1.keys()), set(metrics2.keys()))


class TestProfilingAccuracy(unittest.TestCase):
    """Test profiling accuracy and edge cases"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.profiler = PerformanceProfiler(output_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_zero_memory_growth(self):
        """Test handling of functions with no memory growth"""
        def no_memory_function():
            x = 42
            return x * 2

        metrics = self.profiler.profile_memory(no_memory_function)

        self.assertGreaterEqual(metrics['delta_mb'], -1)  # Allow small variations

    def test_fast_function(self):
        """Test profiling very fast functions"""
        def fast_function():
            return 1 + 1

        metrics = self.profiler.profile_cpu(fast_function)

        self.assertIn('duration', metrics)
        self.assertGreaterEqual(metrics['duration'], 0)

    def test_exception_handling(self):
        """Test profiling function that raises exception"""
        def failing_function():
            raise ValueError("Test exception")

        with self.assertRaises(ValueError):
            self.profiler.profile_cpu(failing_function)


class TestProfilingOutput(unittest.TestCase):
    """Test profiling output formats and files"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.profiler = PerformanceProfiler(output_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_output_directory_creation(self):
        """Test output directory is created"""
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertTrue(os.path.isdir(self.temp_dir))

    def test_profile_stats_file(self):
        """Test CPU profile stats file is created"""
        def test_func():
            return sum(range(1000))

        metrics = self.profiler.profile_cpu(test_func)

        profile_file = metrics['profile_file']
        self.assertTrue(os.path.exists(profile_file))

    def test_report_json_format(self):
        """Test report is valid JSON"""
        import json

        self.profiler.profile_processes()
        report_file = self.profiler.generate_report()

        with open(report_file) as f:
            data = json.load(f)

        self.assertIn('timestamp', data)
        self.assertIn('cpu', data)
        self.assertIn('memory', data)
        self.assertIn('processes', data)


def run_performance_profiler_tests():
    """Run all profiler tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceProfiler))
    suite.addTests(loader.loadTestsFromTestCase(TestProfilingAccuracy))
    suite.addTests(loader.loadTestsFromTestCase(TestProfilingOutput))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_performance_profiler_tests()
    sys.exit(0 if success else 1)
