#!/usr/bin/env python3
"""
OpenTelemetry SLO Monitor
Monitor Service Level Objectives (SLOs) using OpenTelemetry metrics
"""

import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from enum import Enum


class SLOStatus(Enum):
    """SLO compliance status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SLOTarget:
    """Service Level Objective target definition"""
    name: str
    description: str
    target_percentage: float  # e.g., 99.9 for 99.9%
    window_seconds: int  # Time window for SLO calculation
    metric_name: str  # OpenTelemetry metric to monitor
    threshold_value: Optional[float] = None  # For latency SLOs


@dataclass
class SLOResult:
    """SLO monitoring result"""
    slo_name: str
    current_percentage: float
    target_percentage: float
    status: SLOStatus
    error_budget_remaining: float  # Percentage of error budget left
    total_requests: int
    successful_requests: int
    failed_requests: int
    timestamp: float


class SLOMonitor:
    """Monitor Service Level Objectives with OpenTelemetry metrics"""

    def __init__(self):
        self.slos: Dict[str, SLOTarget] = {}
        self.results: List[SLOResult] = []

    def register_slo(self, slo: SLOTarget):
        """
        Register a new SLO to monitor

        Args:
            slo: SLO target definition
        """
        self.slos[slo.name] = slo
        print(f"✅ Registered SLO: {slo.name} (target: {slo.target_percentage}%)")

    def calculate_availability_slo(
        self,
        slo_name: str,
        total_requests: int,
        failed_requests: int
    ) -> SLOResult:
        """
        Calculate availability SLO (e.g., 99.9% uptime)

        Args:
            slo_name: Name of the SLO
            total_requests: Total number of requests
            failed_requests: Number of failed requests

        Returns:
            SLOResult with current status
        """
        if slo_name not in self.slos:
            raise ValueError(f"SLO '{slo_name}' not registered")

        slo = self.slos[slo_name]
        successful_requests = total_requests - failed_requests

        # Calculate current availability percentage
        current_percentage = (
            (successful_requests / total_requests) * 100 if total_requests > 0 else 100.0
        )

        # Calculate error budget
        allowed_failures = total_requests * (1 - slo.target_percentage / 100)
        error_budget_remaining = (
            ((allowed_failures - failed_requests) / allowed_failures) * 100
            if allowed_failures > 0
            else 100.0
        )

        # Determine status
        if current_percentage >= slo.target_percentage:
            status = SLOStatus.HEALTHY
        elif error_budget_remaining > 10:
            status = SLOStatus.WARNING
        else:
            status = SLOStatus.CRITICAL

        result = SLOResult(
            slo_name=slo_name,
            current_percentage=current_percentage,
            target_percentage=slo.target_percentage,
            status=status,
            error_budget_remaining=max(0, error_budget_remaining),
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            timestamp=time.time(),
        )

        self.results.append(result)
        return result

    def calculate_latency_slo(
        self,
        slo_name: str,
        request_latencies: List[float]
    ) -> SLOResult:
        """
        Calculate latency SLO (e.g., 99% of requests < 500ms)

        Args:
            slo_name: Name of the SLO
            request_latencies: List of request latencies in milliseconds

        Returns:
            SLOResult with current status
        """
        if slo_name not in self.slos:
            raise ValueError(f"SLO '{slo_name}' not registered")

        slo = self.slos[slo_name]
        total_requests = len(request_latencies)

        # Count requests meeting latency threshold
        successful_requests = sum(
            1 for latency in request_latencies
            if latency <= slo.threshold_value
        )

        failed_requests = total_requests - successful_requests

        # Calculate current percentage
        current_percentage = (
            (successful_requests / total_requests) * 100 if total_requests > 0 else 100.0
        )

        # Calculate error budget
        allowed_failures = total_requests * (1 - slo.target_percentage / 100)
        error_budget_remaining = (
            ((allowed_failures - failed_requests) / allowed_failures) * 100
            if allowed_failures > 0
            else 100.0
        )

        # Determine status
        if current_percentage >= slo.target_percentage:
            status = SLOStatus.HEALTHY
        elif error_budget_remaining > 10:
            status = SLOStatus.WARNING
        else:
            status = SLOStatus.CRITICAL

        result = SLOResult(
            slo_name=slo_name,
            current_percentage=current_percentage,
            target_percentage=slo.target_percentage,
            status=status,
            error_budget_remaining=max(0, error_budget_remaining),
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            timestamp=time.time(),
        )

        self.results.append(result)
        return result

    def print_report(self, result: SLOResult):
        """Print SLO monitoring report"""
        status_emoji = {
            SLOStatus.HEALTHY: "✅",
            SLOStatus.WARNING: "⚠️",
            SLOStatus.CRITICAL: "🚨",
        }

        print("\n" + "=" * 60)
        print(f"SLO REPORT: {result.slo_name}")
        print("=" * 60)
        print(f"Status:                {status_emoji[result.status]} {result.status.value.upper()}")
        print(f"Current Achievement:   {result.current_percentage:.4f}%")
        print(f"Target:                {result.target_percentage}%")
        print(f"Error Budget Remaining: {result.error_budget_remaining:.2f}%")
        print(f"\nRequests:")
        print(f"  Total:               {result.total_requests}")
        print(f"  Successful:          {result.successful_requests}")
        print(f"  Failed:              {result.failed_requests}")
        print("=" * 60)

        # Recommendations
        if result.status == SLOStatus.CRITICAL:
            print("\n🚨 CRITICAL: SLO violated! Immediate action required.")
            print("   - Investigate recent failures")
            print("   - Check system health and resource usage")
            print("   - Review recent deployments")
        elif result.status == SLOStatus.WARNING:
            print("\n⚠️  WARNING: Error budget depleting.")
            print("   - Monitor closely for further degradation")
            print("   - Consider pausing risky deployments")

    def export_results_json(self, filename: str = "slo-results.json"):
        """Export SLO results to JSON file"""
        data = [
            {**asdict(result), "status": result.status.value}
            for result in self.results
        ]

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n📊 Results exported to: {filename}")


def main():
    """Demo SLO monitoring with sample data"""
    print("OpenTelemetry SLO Monitor - Demo\n")

    monitor = SLOMonitor()

    # Register SLOs
    monitor.register_slo(SLOTarget(
        name="api_availability",
        description="99.9% API availability",
        target_percentage=99.9,
        window_seconds=86400,  # 24 hours
        metric_name="http.server.requests",
    ))

    monitor.register_slo(SLOTarget(
        name="api_latency_p99",
        description="99% of requests < 500ms",
        target_percentage=99.0,
        window_seconds=3600,  # 1 hour
        metric_name="http.server.duration",
        threshold_value=500.0,  # 500ms
    ))

    # Simulate availability SLO monitoring
    print("\n--- Availability SLO Monitoring ---")

    # Scenario 1: Healthy
    result1 = monitor.calculate_availability_slo(
        slo_name="api_availability",
        total_requests=10000,
        failed_requests=5  # 99.95% availability
    )
    monitor.print_report(result1)

    # Scenario 2: Warning
    result2 = monitor.calculate_availability_slo(
        slo_name="api_availability",
        total_requests=10000,
        failed_requests=15  # 99.85% availability
    )
    monitor.print_report(result2)

    # Simulate latency SLO monitoring
    print("\n--- Latency SLO Monitoring ---")

    # Generate sample latencies (milliseconds)
    import random
    latencies = [random.uniform(50, 800) for _ in range(1000)]

    result3 = monitor.calculate_latency_slo(
        slo_name="api_latency_p99",
        request_latencies=latencies
    )
    monitor.print_report(result3)

    # Export results
    monitor.export_results_json()

    print("\n✅ SLO monitoring demo complete!")


if __name__ == "__main__":
    main()
