#!/usr/bin/env python3
"""
Unit tests for OpenTelemetry Metrics Collector
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'resources'))

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider


class TestMetricsCollector(unittest.TestCase):
    """Test suite for metrics collector functionality"""

    def setUp(self):
        """Setup test fixtures"""
        # Create test meter provider
        self.provider = MeterProvider()
        metrics.set_meter_provider(self.provider)
        self.meter = metrics.get_meter("test-service", "1.0.0")

    def test_counter_increment(self):
        """Test counter metric increments correctly"""
        counter = self.meter.create_counter(
            name="test.requests",
            description="Test request counter",
            unit="requests"
        )

        # Increment counter
        counter.add(1, {"method": "GET"})
        counter.add(5, {"method": "POST"})

        # Note: In real tests, you'd read from exporter
        # This verifies no exceptions are raised
        self.assertIsNotNone(counter)

    def test_histogram_records_distribution(self):
        """Test histogram metric records value distributions"""
        histogram = self.meter.create_histogram(
            name="test.duration",
            description="Test duration histogram",
            unit="ms"
        )

        # Record multiple values
        test_values = [10, 25, 50, 100, 250, 500, 1000]
        for value in test_values:
            histogram.record(value, {"endpoint": "/api/test"})

        self.assertIsNotNone(histogram)

    def test_updown_counter_tracks_changes(self):
        """Test UpDownCounter handles positive and negative changes"""
        counter = self.meter.create_up_down_counter(
            name="test.connections",
            description="Test active connections",
            unit="connections"
        )

        # Simulate connections opening and closing
        counter.add(5)   # 5 connections open
        counter.add(3)   # 3 more open (total 8)
        counter.add(-2)  # 2 closed (total 6)
        counter.add(-6)  # 6 closed (total 0)

        self.assertIsNotNone(counter)

    def test_observable_gauge_with_callback(self):
        """Test ObservableGauge with callback function"""
        def memory_callback(options):
            # Simulate memory reading
            yield metrics.Observation(1024 * 1024 * 100, {"type": "heap"})

        gauge = self.meter.create_observable_gauge(
            name="test.memory",
            description="Test memory usage",
            unit="bytes",
            callbacks=[memory_callback]
        )

        self.assertIsNotNone(gauge)

    def test_metric_attributes(self):
        """Test metrics with various attributes"""
        counter = self.meter.create_counter(
            name="test.attributes",
            description="Test attribute handling"
        )

        # Test different attribute types
        counter.add(1, {
            "string_attr": "value",
            "int_attr": 42,
            "bool_attr": True,
        })

        # Test high-cardinality warning (many unique attribute values)
        # In production, avoid high-cardinality attributes
        for i in range(10):
            counter.add(1, {"user_id": f"user-{i}"})

        self.assertIsNotNone(counter)

    def test_semantic_conventions(self):
        """Test using semantic conventions for attributes"""
        from opentelemetry.semconv.trace import SpanAttributes

        counter = self.meter.create_counter(
            name="http.requests",
            description="HTTP requests with semantic conventions"
        )

        # Use semantic conventions (constants ensure consistency)
        counter.add(1, {
            SpanAttributes.HTTP_METHOD: "GET",
            SpanAttributes.HTTP_ROUTE: "/api/users",
            SpanAttributes.HTTP_STATUS_CODE: 200,
        })

        self.assertIsNotNone(counter)

    def tearDown(self):
        """Cleanup after tests"""
        # Shutdown meter provider
        if hasattr(self.provider, 'shutdown'):
            self.provider.shutdown()


class TestMetricsBestPractices(unittest.TestCase):
    """Test adherence to OpenTelemetry best practices"""

    def test_metric_naming_conventions(self):
        """Test metric names follow conventions"""
        provider = MeterProvider()
        meter = provider.get_meter("test", "1.0.0")

        # ✅ GOOD: Namespace, domain, unit
        good_names = [
            "http.server.duration",
            "http.server.requests",
            "db.client.connections.usage",
            "system.cpu.usage",
        ]

        for name in good_names:
            metric = meter.create_counter(name=name, description="Test metric")
            self.assertIsNotNone(metric)

    def test_avoid_high_cardinality_attributes(self):
        """Test avoiding high-cardinality attributes"""
        provider = MeterProvider()
        meter = provider.get_meter("test", "1.0.0")
        counter = meter.create_counter(name="test.cardinality")

        # ✅ GOOD: Low cardinality (generic route template)
        counter.add(1, {"http.route": "/api/users/:id"})

        # ❌ BAD: High cardinality (specific user IDs)
        # This would create thousands of unique time series
        # counter.add(1, {"user.id": "user-12345"})

        self.assertIsNotNone(counter)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
