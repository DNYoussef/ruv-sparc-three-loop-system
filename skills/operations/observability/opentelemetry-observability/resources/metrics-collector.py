#!/usr/bin/env python3
"""
OpenTelemetry Metrics Collector
Automated metrics collection with custom instruments and exporters
"""

import time
import random
from typing import Dict, List, Optional
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    ConsoleMetricExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes


class MetricsCollector:
    """Comprehensive metrics collection for OpenTelemetry observability"""

    def __init__(
        self,
        service_name: str,
        service_version: str = "1.0.0",
        environment: str = "production",
        export_interval_ms: int = 60000,
    ):
        """
        Initialize metrics collector with OpenTelemetry SDK

        Args:
            service_name: Name of the service
            service_version: Version of the service
            environment: Deployment environment (production, staging, dev)
            export_interval_ms: Metrics export interval in milliseconds
        """
        # Create resource with semantic conventions
        resource = Resource.create(
            {
                ResourceAttributes.SERVICE_NAME: service_name,
                ResourceAttributes.SERVICE_VERSION: service_version,
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: environment,
            }
        )

        # Setup metric reader with console exporter (replace with OTLP in production)
        reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(), export_interval_millis=export_interval_ms
        )

        # Initialize MeterProvider
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)

        # Get meter for this service
        self.meter = metrics.get_meter(service_name, service_version)

        # Initialize metric instruments
        self._setup_instruments()

    def _setup_instruments(self):
        """Setup all metric instruments (counters, histograms, gauges)"""

        # COUNTER: Monotonically increasing values
        self.request_counter = self.meter.create_counter(
            name="http.server.requests",
            description="Total HTTP requests received",
            unit="requests",
        )

        self.error_counter = self.meter.create_counter(
            name="http.server.errors",
            description="Total HTTP errors encountered",
            unit="errors",
        )

        # HISTOGRAM: Statistical distributions
        self.request_duration = self.meter.create_histogram(
            name="http.server.duration",
            description="HTTP request duration",
            unit="ms",
        )

        self.payload_size = self.meter.create_histogram(
            name="http.server.payload_size",
            description="HTTP request/response payload size",
            unit="bytes",
        )

        # UPDOWNCOUNTER: Values that can increase or decrease
        self.active_connections = self.meter.create_up_down_counter(
            name="http.server.active_connections",
            description="Number of active HTTP connections",
            unit="connections",
        )

        # OBSERVABLE GAUGE: Async measurements
        self.cpu_usage = self.meter.create_observable_gauge(
            name="system.cpu.usage",
            description="CPU usage percentage",
            unit="%",
            callbacks=[self._get_cpu_usage],
        )

        self.memory_usage = self.meter.create_observable_gauge(
            name="system.memory.usage",
            description="Memory usage in bytes",
            unit="bytes",
            callbacks=[self._get_memory_usage],
        )

    def _get_cpu_usage(self, options):
        """Callback for CPU usage gauge (simulated)"""
        # In production, use psutil.cpu_percent()
        cpu_percent = random.uniform(10, 90)
        yield metrics.Observation(cpu_percent, {"cpu.type": "user"})

    def _get_memory_usage(self, options):
        """Callback for memory usage gauge (simulated)"""
        # In production, use psutil.virtual_memory()
        memory_bytes = random.randint(500_000_000, 2_000_000_000)
        yield metrics.Observation(memory_bytes, {"memory.type": "heap"})

    def record_request(
        self,
        method: str,
        route: str,
        status_code: int,
        duration_ms: float,
        payload_bytes: int = 0,
    ):
        """
        Record HTTP request metrics

        Args:
            method: HTTP method (GET, POST, etc.)
            route: API route template
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
            payload_bytes: Request/response payload size
        """
        attributes = {
            "http.method": method,
            "http.route": route,
            "http.status_code": status_code,
        }

        # Increment request counter
        self.request_counter.add(1, attributes)

        # Record request duration
        self.request_duration.record(duration_ms, attributes)

        # Record payload size if provided
        if payload_bytes > 0:
            self.payload_size.record(payload_bytes, attributes)

        # Record errors (status codes >= 400)
        if status_code >= 400:
            self.error_counter.add(1, attributes)

    def track_connection(self, delta: int):
        """
        Track active connection changes

        Args:
            delta: +1 for new connection, -1 for closed connection
        """
        self.active_connections.add(delta, {"connection.state": "active"})

    def simulate_traffic(self, duration_seconds: int = 60):
        """
        Simulate HTTP traffic for testing metrics collection

        Args:
            duration_seconds: Duration to simulate traffic
        """
        print(f"Simulating traffic for {duration_seconds} seconds...")

        routes = ["/api/users", "/api/orders", "/api/products", "/api/payments"]
        methods = ["GET", "POST", "PUT", "DELETE"]
        status_codes = [200, 201, 400, 404, 500]

        start_time = time.time()
        request_count = 0

        while time.time() - start_time < duration_seconds:
            # Simulate request
            method = random.choice(methods)
            route = random.choice(routes)
            status_code = random.choices(
                status_codes, weights=[70, 15, 5, 5, 5], k=1
            )[0]
            duration_ms = random.uniform(10, 500)
            payload_bytes = random.randint(100, 10000)

            # Track connection open
            self.track_connection(1)

            # Record request metrics
            self.record_request(method, route, status_code, duration_ms, payload_bytes)

            # Track connection close
            self.track_connection(-1)

            request_count += 1

            # Random delay between requests
            time.sleep(random.uniform(0.1, 1.0))

        print(f"Simulation complete. Processed {request_count} requests.")


def main():
    """Main entry point for metrics collector"""

    # Initialize metrics collector
    collector = MetricsCollector(
        service_name="api-gateway",
        service_version="2.0.0",
        environment="production",
        export_interval_ms=10000,  # Export every 10 seconds for demo
    )

    # Simulate traffic for 30 seconds
    collector.simulate_traffic(duration_seconds=30)

    # Keep alive to allow final metric export
    print("Waiting for final metric export...")
    time.sleep(15)

    print("Metrics collection complete!")


if __name__ == "__main__":
    main()
