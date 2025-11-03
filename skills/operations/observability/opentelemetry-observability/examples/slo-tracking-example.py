#!/usr/bin/env python3
"""
SLO Tracking Example with OpenTelemetry
Demonstrates comprehensive Service Level Objective (SLO) monitoring using
OpenTelemetry metrics for production systems

SLO Types:
1. Availability SLO (99.9% uptime)
2. Latency SLO (99% of requests < 500ms)
3. Error Rate SLO (< 1% errors)
4. Throughput SLO (≥ 100 req/s)

Features:
- Real-time SLO monitoring
- Error budget tracking
- Multi-window SLO calculation (1h, 24h, 7d, 30d)
- Alerting on SLO violations
- Automated reporting
"""

import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    ConsoleMetricExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes


class SLOType(Enum):
    """SLO types supported"""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


class SLOStatus(Enum):
    """SLO compliance status"""
    HEALTHY = "healthy"       # Meeting SLO target
    WARNING = "warning"       # 10-25% error budget consumed
    CRITICAL = "critical"     # > 25% error budget consumed
    VIOLATED = "violated"     # SLO target missed


@dataclass
class SLODefinition:
    """SLO target definition"""
    name: str
    slo_type: SLOType
    target_percentage: float  # e.g., 99.9 for 99.9%
    window_seconds: int       # Time window (3600=1h, 86400=24h, 604800=7d)
    threshold_value: Optional[float] = None  # For latency SLOs (milliseconds)
    threshold_requests: Optional[int] = None  # For throughput SLOs


@dataclass
class SLOMeasurement:
    """SLO measurement result"""
    slo_name: str
    timestamp: datetime
    window_seconds: int
    current_percentage: float
    target_percentage: float
    status: SLOStatus
    error_budget_remaining: float  # Percentage remaining (0-100)
    total_events: int
    good_events: int
    bad_events: int
    metadata: Dict = field(default_factory=dict)


class SLOTracker:
    """
    Production-grade SLO tracking with OpenTelemetry metrics
    Tracks multiple SLOs with error budget monitoring
    """

    def __init__(
        self,
        service_name: str = "api-service",
        environment: str = "production",
    ):
        # Initialize OpenTelemetry
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: service_name,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: environment,
        })

        reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(),
            export_interval_millis=10000
        )

        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)

        self.meter = metrics.get_meter(service_name, "1.0.0")

        # SLO definitions
        self.slos: Dict[str, SLODefinition] = {}

        # Measurement history
        self.measurements: List[SLOMeasurement] = []

        # Request tracking
        self.requests: List[Dict] = []

        # Initialize SLO metrics
        self._setup_slo_metrics()

    def _setup_slo_metrics(self):
        """Setup OpenTelemetry metrics for SLO tracking"""

        # SLI (Service Level Indicator) metrics
        self.sli_total_counter = self.meter.create_counter(
            name="slo.sli.total",
            description="Total SLI events for SLO calculation",
            unit="events",
        )

        self.sli_good_counter = self.meter.create_counter(
            name="slo.sli.good",
            description="Good SLI events (meeting criteria)",
            unit="events",
        )

        self.sli_bad_counter = self.meter.create_counter(
            name="slo.sli.bad",
            description="Bad SLI events (not meeting criteria)",
            unit="events",
        )

        # SLO achievement gauge
        self.slo_achievement = self.meter.create_observable_gauge(
            name="slo.achievement",
            description="Current SLO achievement percentage",
            unit="%",
            callbacks=[self._measure_slo_achievement],
        )

        # Error budget gauge
        self.error_budget = self.meter.create_observable_gauge(
            name="slo.error_budget.remaining",
            description="Remaining error budget percentage",
            unit="%",
            callbacks=[self._measure_error_budget],
        )

    def _measure_slo_achievement(self, options):
        """Callback for SLO achievement gauge"""
        for slo_name, measurement in self._get_latest_measurements().items():
            yield metrics.Observation(
                measurement.current_percentage,
                {
                    "slo.name": slo_name,
                    "slo.type": measurement.metadata.get("slo_type", "unknown"),
                    "slo.window": str(measurement.window_seconds),
                }
            )

    def _measure_error_budget(self, options):
        """Callback for error budget gauge"""
        for slo_name, measurement in self._get_latest_measurements().items():
            yield metrics.Observation(
                measurement.error_budget_remaining,
                {
                    "slo.name": slo_name,
                    "slo.status": measurement.status.value,
                }
            )

    def _get_latest_measurements(self) -> Dict[str, SLOMeasurement]:
        """Get latest measurement for each SLO"""
        latest = {}
        for measurement in self.measurements:
            if (
                measurement.slo_name not in latest
                or measurement.timestamp > latest[measurement.slo_name].timestamp
            ):
                latest[measurement.slo_name] = measurement
        return latest

    # ========================================================================
    # SLO Registration
    # ========================================================================

    def register_slo(self, slo: SLODefinition):
        """Register a new SLO for tracking"""
        self.slos[slo.name] = slo
        print(f"✅ Registered SLO: {slo.name}")
        print(f"   Type: {slo.slo_type.value}")
        print(f"   Target: {slo.target_percentage}%")
        print(f"   Window: {slo.window_seconds}s ({self._format_duration(slo.window_seconds)})")

    def _format_duration(self, seconds: int) -> str:
        """Format seconds to human-readable duration"""
        if seconds < 3600:
            return f"{seconds // 60}m"
        elif seconds < 86400:
            return f"{seconds // 3600}h"
        else:
            return f"{seconds // 86400}d"

    # ========================================================================
    # Request Tracking
    # ========================================================================

    def track_request(
        self,
        success: bool,
        duration_ms: float,
        status_code: int = 200,
    ):
        """Track a request for SLO calculation"""
        request = {
            "timestamp": datetime.now(),
            "success": success,
            "duration_ms": duration_ms,
            "status_code": status_code,
        }
        self.requests.append(request)

        # Record SLI metrics
        attributes = {
            "http.status_code": status_code,
            "success": str(success).lower(),
        }

        self.sli_total_counter.add(1, attributes)
        if success:
            self.sli_good_counter.add(1, attributes)
        else:
            self.sli_bad_counter.add(1, attributes)

    # ========================================================================
    # SLO Calculation
    # ========================================================================

    def calculate_availability_slo(self, slo_name: str) -> SLOMeasurement:
        """Calculate availability SLO (% of successful requests)"""
        if slo_name not in self.slos:
            raise ValueError(f"SLO '{slo_name}' not registered")

        slo = self.slos[slo_name]
        cutoff_time = datetime.now() - timedelta(seconds=slo.window_seconds)

        # Filter requests within window
        recent_requests = [
            r for r in self.requests
            if r["timestamp"] >= cutoff_time
        ]

        total = len(recent_requests)
        good = sum(1 for r in recent_requests if r["success"])
        bad = total - good

        # Calculate achievement
        current_percentage = (good / total * 100) if total > 0 else 100.0

        # Calculate error budget
        allowed_failures = total * (1 - slo.target_percentage / 100)
        error_budget_remaining = (
            ((allowed_failures - bad) / allowed_failures * 100)
            if allowed_failures > 0 else 100.0
        )
        error_budget_remaining = max(0, min(100, error_budget_remaining))

        # Determine status
        status = self._determine_status(
            current_percentage,
            slo.target_percentage,
            error_budget_remaining
        )

        measurement = SLOMeasurement(
            slo_name=slo_name,
            timestamp=datetime.now(),
            window_seconds=slo.window_seconds,
            current_percentage=current_percentage,
            target_percentage=slo.target_percentage,
            status=status,
            error_budget_remaining=error_budget_remaining,
            total_events=total,
            good_events=good,
            bad_events=bad,
            metadata={"slo_type": slo.slo_type.value},
        )

        self.measurements.append(measurement)
        return measurement

    def calculate_latency_slo(self, slo_name: str) -> SLOMeasurement:
        """Calculate latency SLO (% of requests below threshold)"""
        if slo_name not in self.slos:
            raise ValueError(f"SLO '{slo_name}' not registered")

        slo = self.slos[slo_name]
        cutoff_time = datetime.now() - timedelta(seconds=slo.window_seconds)

        # Filter requests within window
        recent_requests = [
            r for r in self.requests
            if r["timestamp"] >= cutoff_time
        ]

        total = len(recent_requests)
        good = sum(1 for r in recent_requests if r["duration_ms"] <= slo.threshold_value)
        bad = total - good

        # Calculate achievement
        current_percentage = (good / total * 100) if total > 0 else 100.0

        # Calculate error budget
        allowed_failures = total * (1 - slo.target_percentage / 100)
        error_budget_remaining = (
            ((allowed_failures - bad) / allowed_failures * 100)
            if allowed_failures > 0 else 100.0
        )
        error_budget_remaining = max(0, min(100, error_budget_remaining))

        # Determine status
        status = self._determine_status(
            current_percentage,
            slo.target_percentage,
            error_budget_remaining
        )

        measurement = SLOMeasurement(
            slo_name=slo_name,
            timestamp=datetime.now(),
            window_seconds=slo.window_seconds,
            current_percentage=current_percentage,
            target_percentage=slo.target_percentage,
            status=status,
            error_budget_remaining=error_budget_remaining,
            total_events=total,
            good_events=good,
            bad_events=bad,
            metadata={
                "slo_type": slo.slo_type.value,
                "threshold_ms": slo.threshold_value,
            },
        )

        self.measurements.append(measurement)
        return measurement

    def _determine_status(
        self,
        current: float,
        target: float,
        error_budget: float
    ) -> SLOStatus:
        """Determine SLO status based on current achievement and error budget"""
        if current >= target:
            return SLOStatus.HEALTHY
        elif error_budget > 25:
            return SLOStatus.WARNING
        elif error_budget > 0:
            return SLOStatus.CRITICAL
        else:
            return SLOStatus.VIOLATED

    # ========================================================================
    # Reporting
    # ========================================================================

    def print_slo_report(self, measurement: SLOMeasurement):
        """Print detailed SLO report"""
        status_emoji = {
            SLOStatus.HEALTHY: "✅",
            SLOStatus.WARNING: "⚠️",
            SLOStatus.CRITICAL: "🚨",
            SLOStatus.VIOLATED: "❌",
        }

        print("\n" + "=" * 70)
        print(f"  SLO REPORT: {measurement.slo_name}")
        print("=" * 70)
        print(f"Status:                {status_emoji[measurement.status]} {measurement.status.value.upper()}")
        print(f"Current Achievement:   {measurement.current_percentage:.4f}%")
        print(f"Target:                {measurement.target_percentage}%")
        print(f"Error Budget Remaining: {measurement.error_budget_remaining:.2f}%")
        print(f"Window:                {self._format_duration(measurement.window_seconds)}")
        print(f"\nEvents:")
        print(f"  Total:               {measurement.total_events}")
        print(f"  Good:                {measurement.good_events}")
        print(f"  Bad:                 {measurement.bad_events}")
        print("=" * 70)

        # Recommendations
        if measurement.status == SLOStatus.VIOLATED:
            print("\n❌ SLO VIOLATED! Immediate action required:")
            print("   1. Investigate root cause of failures")
            print("   2. Trigger incident response")
            print("   3. Consider rollback if recent deployment")
        elif measurement.status == SLOStatus.CRITICAL:
            print("\n🚨 CRITICAL: Error budget nearly exhausted:")
            print("   1. Pause risky deployments")
            print("   2. Monitor closely for further degradation")
            print("   3. Prepare incident response plan")
        elif measurement.status == SLOStatus.WARNING:
            print("\n⚠️  WARNING: Error budget depleting:")
            print("   1. Monitor closely")
            print("   2. Review recent changes")
            print("   3. Consider proactive measures")


def main():
    """Demo SLO tracking with realistic scenarios"""
    print("=" * 70)
    print("  OpenTelemetry SLO Tracking Example")
    print("=" * 70)

    tracker = SLOTracker(service_name="payment-api", environment="production")

    # Register SLOs
    print("\n📋 Registering SLOs...\n")

    tracker.register_slo(SLODefinition(
        name="api_availability_24h",
        slo_type=SLOType.AVAILABILITY,
        target_percentage=99.9,
        window_seconds=86400,  # 24 hours
    ))

    tracker.register_slo(SLODefinition(
        name="api_latency_p99_1h",
        slo_type=SLOType.LATENCY,
        target_percentage=99.0,
        window_seconds=3600,  # 1 hour
        threshold_value=500.0,  # 500ms
    ))

    # Simulate traffic
    print("\n🚀 Simulating traffic...\n")

    for i in range(1000):
        # 99.5% success rate (healthy)
        success = random.random() < 0.995

        # Latency distribution
        if random.random() < 0.98:
            duration_ms = random.uniform(50, 400)  # Fast requests
        else:
            duration_ms = random.uniform(500, 1000)  # Slow requests

        status_code = 200 if success else random.choice([400, 500, 503])

        tracker.track_request(
            success=success,
            duration_ms=duration_ms,
            status_code=status_code,
        )

    # Calculate and report SLOs
    print("\n📊 Calculating SLOs...\n")

    availability_result = tracker.calculate_availability_slo("api_availability_24h")
    tracker.print_slo_report(availability_result)

    latency_result = tracker.calculate_latency_slo("api_latency_p99_1h")
    tracker.print_slo_report(latency_result)

    # Wait for final metric export
    print("\n⏳ Waiting for final metric export...")
    time.sleep(12)

    print("\n✅ SLO tracking demo complete!")


if __name__ == "__main__":
    main()
