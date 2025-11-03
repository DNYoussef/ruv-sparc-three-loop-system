#!/usr/bin/env python3
"""
Metrics Monitoring Example with OpenTelemetry
Demonstrates comprehensive metrics collection, aggregation, and monitoring
for production applications

Metrics Categories:
1. Request metrics (counters, histograms)
2. System metrics (CPU, memory, disk)
3. Business metrics (orders, revenue, users)
4. SLI/SLO metrics (availability, latency, error rate)
"""

import time
import random
import threading
from typing import Dict, List
from dataclasses import dataclass

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    ConsoleMetricExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.semconv.trace import SpanAttributes


@dataclass
class RequestMetrics:
    """Request-level metrics tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration_ms: float = 0.0


class ProductionMetricsMonitor:
    """
    Production-grade metrics monitoring with OpenTelemetry
    Tracks request, system, and business metrics
    """

    def __init__(
        self,
        service_name: str = "ecommerce-api",
        service_version: str = "2.0.0",
        environment: str = "production",
    ):
        # Initialize OpenTelemetry Metrics SDK
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: service_name,
            ResourceAttributes.SERVICE_VERSION: service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: environment,
        })

        # Create metric reader with console exporter (use OTLP in production)
        reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(),
            export_interval_millis=10000  # Export every 10 seconds
        )

        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)

        self.meter = metrics.get_meter(service_name, service_version)
        self.request_stats = RequestMetrics()

        # Initialize metric instruments
        self._setup_request_metrics()
        self._setup_system_metrics()
        self._setup_business_metrics()
        self._setup_sli_metrics()

    def _setup_request_metrics(self):
        """Setup HTTP request metrics"""
        # Counter: Total requests
        self.request_counter = self.meter.create_counter(
            name="http.server.requests",
            description="Total HTTP requests received",
            unit="requests",
        )

        # Histogram: Request duration distribution
        self.request_duration = self.meter.create_histogram(
            name="http.server.duration",
            description="HTTP request duration distribution",
            unit="ms",
        )

        # Counter: Request payload size
        self.payload_size_counter = self.meter.create_counter(
            name="http.server.request.size",
            description="Total bytes received",
            unit="bytes",
        )

        # Counter: Response payload size
        self.response_size_counter = self.meter.create_counter(
            name="http.server.response.size",
            description="Total bytes sent",
            unit="bytes",
        )

        # UpDownCounter: Active requests
        self.active_requests = self.meter.create_up_down_counter(
            name="http.server.active_requests",
            description="Number of requests currently being processed",
            unit="requests",
        )

    def _setup_system_metrics(self):
        """Setup system-level metrics"""
        # ObservableGauge: CPU usage
        self.cpu_usage = self.meter.create_observable_gauge(
            name="system.cpu.utilization",
            description="CPU utilization percentage",
            unit="%",
            callbacks=[self._measure_cpu_usage],
        )

        # ObservableGauge: Memory usage
        self.memory_usage = self.meter.create_observable_gauge(
            name="system.memory.usage",
            description="Memory usage in bytes",
            unit="bytes",
            callbacks=[self._measure_memory_usage],
        )

        # ObservableGauge: Disk usage
        self.disk_usage = self.meter.create_observable_gauge(
            name="system.disk.usage",
            description="Disk usage percentage",
            unit="%",
            callbacks=[self._measure_disk_usage],
        )

    def _setup_business_metrics(self):
        """Setup business/application metrics"""
        # Counter: Orders processed
        self.orders_counter = self.meter.create_counter(
            name="business.orders.processed",
            description="Total orders processed",
            unit="orders",
        )

        # Counter: Revenue
        self.revenue_counter = self.meter.create_counter(
            name="business.revenue.total",
            description="Total revenue in USD",
            unit="USD",
        )

        # UpDownCounter: Active users
        self.active_users = self.meter.create_up_down_counter(
            name="business.users.active",
            description="Number of currently active users",
            unit="users",
        )

        # Histogram: Order value distribution
        self.order_value = self.meter.create_histogram(
            name="business.order.value",
            description="Order value distribution",
            unit="USD",
        )

    def _setup_sli_metrics(self):
        """Setup SLI (Service Level Indicator) metrics for SLO monitoring"""
        # Counter: SLI - Successful requests (for availability)
        self.sli_success_counter = self.meter.create_counter(
            name="sli.requests.successful",
            description="Successful requests for SLI calculation",
            unit="requests",
        )

        # Counter: SLI - Total requests (for availability)
        self.sli_total_counter = self.meter.create_counter(
            name="sli.requests.total",
            description="Total requests for SLI calculation",
            unit="requests",
        )

        # Histogram: SLI - Latency (for latency SLO)
        self.sli_latency = self.meter.create_histogram(
            name="sli.latency",
            description="Request latency for SLI calculation",
            unit="ms",
        )

    # ========================================================================
    # Metric Callbacks (for ObservableGauges)
    # ========================================================================

    def _measure_cpu_usage(self, options):
        """Measure CPU usage (simulated for demo)"""
        # In production, use psutil.cpu_percent()
        cpu_percent = random.uniform(15, 75)
        yield metrics.Observation(cpu_percent, {})

    def _measure_memory_usage(self, options):
        """Measure memory usage (simulated for demo)"""
        # In production, use psutil.virtual_memory()
        memory_bytes = random.randint(500_000_000, 2_000_000_000)
        yield metrics.Observation(memory_bytes, {"memory.type": "heap"})

    def _measure_disk_usage(self, options):
        """Measure disk usage (simulated for demo)"""
        # In production, use psutil.disk_usage('/')
        disk_percent = random.uniform(40, 80)
        yield metrics.Observation(disk_percent, {"disk.device": "/dev/sda1"})

    # ========================================================================
    # Request Tracking
    # ========================================================================

    def track_request(
        self,
        method: str,
        route: str,
        status_code: int,
        duration_ms: float,
        request_size_bytes: int = 0,
        response_size_bytes: int = 0,
    ):
        """
        Track HTTP request metrics

        Args:
            method: HTTP method (GET, POST, etc.)
            route: API route template (e.g., /api/orders/:id)
            status_code: HTTP status code
            duration_ms: Request processing duration in milliseconds
            request_size_bytes: Request payload size in bytes
            response_size_bytes: Response payload size in bytes
        """
        attributes = {
            SpanAttributes.HTTP_METHOD: method,
            SpanAttributes.HTTP_ROUTE: route,
            SpanAttributes.HTTP_STATUS_CODE: status_code,
        }

        # Record request
        self.request_counter.add(1, attributes)
        self.request_duration.record(duration_ms, attributes)

        # Record payload sizes
        if request_size_bytes > 0:
            self.payload_size_counter.add(request_size_bytes, attributes)
        if response_size_bytes > 0:
            self.response_size_counter.add(response_size_bytes, attributes)

        # Update SLI metrics
        self.sli_total_counter.add(1, attributes)
        if 200 <= status_code < 400:
            self.sli_success_counter.add(1, attributes)
        self.sli_latency.record(duration_ms, attributes)

        # Update internal stats
        self.request_stats.total_requests += 1
        if status_code < 400:
            self.request_stats.successful_requests += 1
        else:
            self.request_stats.failed_requests += 1
        self.request_stats.total_duration_ms += duration_ms

    def track_order(self, order_value: float, status: str):
        """Track business metric: order processed"""
        attributes = {"order.status": status}

        self.orders_counter.add(1, attributes)
        self.order_value.record(order_value, attributes)

        if status == "completed":
            self.revenue_counter.add(order_value, {"revenue.type": "sales"})

    def track_user_login(self):
        """Track user login (increment active users)"""
        self.active_users.add(1, {"user.state": "logged_in"})

    def track_user_logout(self):
        """Track user logout (decrement active users)"""
        self.active_users.add(-1, {"user.state": "logged_out"})

    # ========================================================================
    # Simulation & Demo
    # ========================================================================

    def simulate_traffic(self, duration_seconds: int = 60):
        """
        Simulate realistic traffic for demonstration

        Args:
            duration_seconds: Duration to run simulation
        """
        print(f"\n🚀 Starting traffic simulation for {duration_seconds} seconds...\n")

        routes = [
            ("/api/products", "GET"),
            ("/api/products/:id", "GET"),
            ("/api/orders", "POST"),
            ("/api/orders/:id", "GET"),
            ("/api/users/profile", "GET"),
            ("/api/cart", "POST"),
            ("/api/checkout", "POST"),
        ]

        status_codes = [200, 201, 400, 404, 500, 503]
        weights = [60, 15, 10, 8, 5, 2]  # Success rate: ~75%

        start_time = time.time()
        request_count = 0

        while time.time() - start_time < duration_seconds:
            # Simulate request
            route, method = random.choice(routes)
            status_code = random.choices(status_codes, weights=weights)[0]
            duration_ms = random.uniform(10, 500)
            request_size = random.randint(100, 5000)
            response_size = random.randint(500, 20000)

            # Track active request
            self.active_requests.add(1)

            # Process request
            self.track_request(
                method=method,
                route=route,
                status_code=status_code,
                duration_ms=duration_ms,
                request_size_bytes=request_size,
                response_size_bytes=response_size,
            )

            # Track active request completion
            self.active_requests.add(-1)

            # Simulate business events
            if route == "/api/orders" and method == "POST" and status_code in [200, 201]:
                order_value = random.uniform(10.0, 500.0)
                self.track_order(order_value, "completed")

            # Simulate user activity
            if random.random() < 0.1:  # 10% chance
                self.track_user_login()
            if random.random() < 0.05:  # 5% chance
                self.track_user_logout()

            request_count += 1

            # Random delay between requests
            time.sleep(random.uniform(0.05, 0.2))

        print(f"\n✅ Simulation complete! Processed {request_count} requests.")
        self._print_summary()

    def _print_summary(self):
        """Print summary statistics"""
        print("\n" + "=" * 60)
        print("  METRICS SUMMARY")
        print("=" * 60)
        print(f"Total Requests:       {self.request_stats.total_requests}")
        print(f"Successful:           {self.request_stats.successful_requests}")
        print(f"Failed:               {self.request_stats.failed_requests}")

        if self.request_stats.total_requests > 0:
            success_rate = (
                self.request_stats.successful_requests / self.request_stats.total_requests
            ) * 100
            avg_duration = (
                self.request_stats.total_duration_ms / self.request_stats.total_requests
            )
            print(f"Success Rate:         {success_rate:.2f}%")
            print(f"Avg Duration:         {avg_duration:.2f}ms")

        print("=" * 60)


def main():
    """Main entry point for metrics monitoring demo"""
    print("=" * 60)
    print("  OpenTelemetry Metrics Monitoring Example")
    print("=" * 60)

    # Initialize monitor
    monitor = ProductionMetricsMonitor(
        service_name="ecommerce-api",
        service_version="2.0.0",
        environment="production",
    )

    # Run traffic simulation
    monitor.simulate_traffic(duration_seconds=30)

    # Wait for final metric export
    print("\n⏳ Waiting for final metric export...")
    time.sleep(12)

    print("\n✅ Metrics monitoring demo complete!")
    print("\nℹ️  In production, replace ConsoleMetricExporter with OTLP exporter:")
    print("   - Prometheus: OTLPMetricExporter(url='http://localhost:4318/v1/metrics')")
    print("   - Grafana Cloud: Configure OTLP endpoint with API key")


if __name__ == "__main__":
    main()
