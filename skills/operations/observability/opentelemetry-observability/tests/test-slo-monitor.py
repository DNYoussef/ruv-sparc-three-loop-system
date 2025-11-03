#!/usr/bin/env python3
"""
Unit tests for OpenTelemetry SLO Monitor
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'resources'))

from slo_monitor import SLOMonitor, SLOTarget, SLOStatus


class TestSLOMonitor(unittest.TestCase):
    """Test suite for SLO monitoring functionality"""

    def setUp(self):
        """Setup test fixtures"""
        self.monitor = SLOMonitor()

    def test_register_slo(self):
        """Test registering a new SLO"""
        slo = SLOTarget(
            name="test_availability",
            description="Test SLO for availability",
            target_percentage=99.9,
            window_seconds=3600,
            metric_name="test.requests",
        )

        self.monitor.register_slo(slo)
        self.assertIn("test_availability", self.monitor.slos)
        self.assertEqual(self.monitor.slos["test_availability"].target_percentage, 99.9)

    def test_availability_slo_healthy(self):
        """Test availability SLO in healthy state"""
        slo = SLOTarget(
            name="api_availability",
            description="99.9% API availability",
            target_percentage=99.9,
            window_seconds=86400,
            metric_name="http.requests",
        )
        self.monitor.register_slo(slo)

        # Simulate healthy system: 99.95% availability
        result = self.monitor.calculate_availability_slo(
            slo_name="api_availability",
            total_requests=10000,
            failed_requests=5  # 0.05% error rate
        )

        self.assertEqual(result.status, SLOStatus.HEALTHY)
        self.assertGreaterEqual(result.current_percentage, 99.9)
        self.assertGreater(result.error_budget_remaining, 0)

    def test_availability_slo_warning(self):
        """Test availability SLO in warning state"""
        slo = SLOTarget(
            name="api_availability",
            description="99.9% API availability",
            target_percentage=99.9,
            window_seconds=86400,
            metric_name="http.requests",
        )
        self.monitor.register_slo(slo)

        # Simulate degraded system: 99.85% availability
        result = self.monitor.calculate_availability_slo(
            slo_name="api_availability",
            total_requests=10000,
            failed_requests=15  # 0.15% error rate
        )

        # Should be WARNING (below target but error budget > 10%)
        self.assertIn(result.status, [SLOStatus.WARNING, SLOStatus.CRITICAL])
        self.assertLess(result.current_percentage, 99.9)

    def test_availability_slo_critical(self):
        """Test availability SLO in critical state"""
        slo = SLOTarget(
            name="api_availability",
            description="99.9% API availability",
            target_percentage=99.9,
            window_seconds=86400,
            metric_name="http.requests",
        )
        self.monitor.register_slo(slo)

        # Simulate failing system: 99.0% availability
        result = self.monitor.calculate_availability_slo(
            slo_name="api_availability",
            total_requests=10000,
            failed_requests=100  # 1.0% error rate
        )

        self.assertEqual(result.status, SLOStatus.CRITICAL)
        self.assertLess(result.current_percentage, 99.9)
        self.assertLess(result.error_budget_remaining, 10)

    def test_latency_slo_healthy(self):
        """Test latency SLO in healthy state"""
        slo = SLOTarget(
            name="api_latency",
            description="99% of requests < 500ms",
            target_percentage=99.0,
            window_seconds=3600,
            metric_name="http.duration",
            threshold_value=500.0
        )
        self.monitor.register_slo(slo)

        # Simulate healthy latencies (all under 500ms)
        latencies = [100, 150, 200, 250, 300] * 20  # 100 requests
        result = self.monitor.calculate_latency_slo(
            slo_name="api_latency",
            request_latencies=latencies
        )

        self.assertEqual(result.status, SLOStatus.HEALTHY)
        self.assertEqual(result.current_percentage, 100.0)

    def test_latency_slo_with_slow_requests(self):
        """Test latency SLO with some slow requests"""
        slo = SLOTarget(
            name="api_latency",
            description="99% of requests < 500ms",
            target_percentage=99.0,
            window_seconds=3600,
            metric_name="http.duration",
            threshold_value=500.0
        )
        self.monitor.register_slo(slo)

        # 98 fast requests + 2 slow requests = 98% under threshold
        latencies = [200] * 98 + [600, 700]  # 100 requests total
        result = self.monitor.calculate_latency_slo(
            slo_name="api_latency",
            request_latencies=latencies
        )

        self.assertEqual(result.current_percentage, 98.0)
        self.assertEqual(result.failed_requests, 2)
        self.assertIn(result.status, [SLOStatus.WARNING, SLOStatus.CRITICAL])

    def test_error_budget_calculation(self):
        """Test error budget calculation"""
        slo = SLOTarget(
            name="test_slo",
            description="Test error budget",
            target_percentage=99.0,
            window_seconds=3600,
            metric_name="test.metric",
        )
        self.monitor.register_slo(slo)

        # With 99% SLO, we allow 1% errors
        # 10000 requests * 1% = 100 allowed failures
        result = self.monitor.calculate_availability_slo(
            slo_name="test_slo",
            total_requests=10000,
            failed_requests=50  # Used 50 of 100 budget
        )

        # Error budget remaining = (100 - 50) / 100 * 100 = 50%
        self.assertAlmostEqual(result.error_budget_remaining, 50.0, delta=0.1)

    def test_zero_requests(self):
        """Test SLO calculation with zero requests"""
        slo = SLOTarget(
            name="test_slo",
            description="Test with no traffic",
            target_percentage=99.0,
            window_seconds=3600,
            metric_name="test.metric",
        )
        self.monitor.register_slo(slo)

        result = self.monitor.calculate_availability_slo(
            slo_name="test_slo",
            total_requests=0,
            failed_requests=0
        )

        # With no requests, should be 100% (healthy)
        self.assertEqual(result.current_percentage, 100.0)
        self.assertEqual(result.status, SLOStatus.HEALTHY)

    def test_multiple_slo_tracking(self):
        """Test tracking multiple SLOs simultaneously"""
        # Register multiple SLOs
        slo1 = SLOTarget(
            name="slo_1",
            description="First SLO",
            target_percentage=99.9,
            window_seconds=3600,
            metric_name="metric_1",
        )
        slo2 = SLOTarget(
            name="slo_2",
            description="Second SLO",
            target_percentage=99.0,
            window_seconds=3600,
            metric_name="metric_2",
        )

        self.monitor.register_slo(slo1)
        self.monitor.register_slo(slo2)

        # Calculate both SLOs
        result1 = self.monitor.calculate_availability_slo("slo_1", 1000, 1)
        result2 = self.monitor.calculate_availability_slo("slo_2", 1000, 10)

        self.assertEqual(len(self.monitor.results), 2)
        self.assertNotEqual(result1.slo_name, result2.slo_name)

    def test_invalid_slo_name(self):
        """Test error handling for invalid SLO name"""
        with self.assertRaises(ValueError):
            self.monitor.calculate_availability_slo(
                slo_name="nonexistent_slo",
                total_requests=1000,
                failed_requests=10
            )


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
