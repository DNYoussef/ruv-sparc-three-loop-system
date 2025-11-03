#!/usr/bin/env python3
"""
Hive Optimizer - Performance Tuning and Optimization for Hive Mind
Analyzes hive performance metrics and provides optimization recommendations.
"""

import json
import sqlite3
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    metric_name: str
    value: float
    timestamp: str
    worker_id: Optional[str] = None


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation"""
    category: str
    priority: str  # high, medium, low
    issue: str
    recommendation: str
    expected_improvement: str


class HiveOptimizer:
    """Performance optimizer for hive mind systems"""

    def __init__(self, session_file: str = ".hive-mind/session.json",
                 memory_db: str = ".hive-mind/collective-memory.db"):
        self.session_file = Path(session_file)
        self.memory_db = Path(memory_db)
        self.metrics: List[PerformanceMetric] = []
        self.recommendations: List[OptimizationRecommendation] = []

    def load_session_data(self) -> Dict:
        """Load hive mind session data"""
        if not self.session_file.exists():
            raise FileNotFoundError(f"Session file not found: {self.session_file}")

        with open(self.session_file, 'r') as f:
            return json.load(f)

    def analyze_performance(self) -> Dict:
        """Comprehensive performance analysis"""
        session = self.load_session_data()

        analysis = {
            "worker_utilization": self._analyze_worker_utilization(session),
            "task_efficiency": self._analyze_task_efficiency(session),
            "consensus_quality": self._analyze_consensus_quality(session),
            "memory_health": self._analyze_memory_health(),
            "bottlenecks": self._identify_bottlenecks(session),
            "timestamp": datetime.utcnow().isoformat()
        }

        # Generate recommendations based on analysis
        self._generate_recommendations(analysis)

        return analysis

    def _analyze_worker_utilization(self, session: Dict) -> Dict:
        """Analyze worker utilization patterns"""
        workers = session.get('workers', [])
        if not workers:
            return {"status": "no_workers"}

        total_workers = len(workers)
        idle_workers = sum(1 for w in workers if w.get('status') == 'idle')
        busy_workers = sum(1 for w in workers if w.get('status') == 'busy')

        # Calculate utilization rate
        utilization_rate = busy_workers / total_workers if total_workers > 0 else 0

        # Analyze task completion rates
        completion_counts = [w.get('tasks_completed', 0) for w in workers]
        avg_completion = statistics.mean(completion_counts) if completion_counts else 0
        completion_variance = statistics.variance(completion_counts) if len(completion_counts) > 1 else 0

        return {
            "total_workers": total_workers,
            "idle_workers": idle_workers,
            "busy_workers": busy_workers,
            "utilization_rate": round(utilization_rate, 3),
            "avg_tasks_completed": round(avg_completion, 2),
            "completion_variance": round(completion_variance, 2),
            "load_balance_score": self._calculate_load_balance(completion_variance, avg_completion)
        }

    def _calculate_load_balance(self, variance: float, mean: float) -> str:
        """Calculate load balance quality"""
        if mean == 0:
            return "no_data"

        cv = (variance ** 0.5) / mean  # Coefficient of variation

        if cv < 0.2:
            return "excellent"
        elif cv < 0.4:
            return "good"
        elif cv < 0.6:
            return "fair"
        else:
            return "poor"

    def _analyze_task_efficiency(self, session: Dict) -> Dict:
        """Analyze task execution efficiency"""
        tasks = session.get('tasks', [])
        if not tasks:
            return {"status": "no_tasks"}

        total_tasks = len(tasks)
        completed = sum(1 for t in tasks if t.get('status') == 'completed')
        in_progress = sum(1 for t in tasks if t.get('status') == 'in_progress')
        pending = sum(1 for t in tasks if t.get('status') == 'pending')

        completion_rate = completed / total_tasks if total_tasks > 0 else 0

        # Analyze task durations
        durations = []
        for task in tasks:
            if task.get('status') == 'completed' and task.get('completed_at') and task.get('created_at'):
                created = datetime.fromisoformat(task['created_at'].replace('Z', '+00:00'))
                completed_at = datetime.fromisoformat(task['completed_at'].replace('Z', '+00:00'))
                duration = (completed_at - created).total_seconds()
                durations.append(duration)

        avg_duration = statistics.mean(durations) if durations else 0

        return {
            "total_tasks": total_tasks,
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "completion_rate": round(completion_rate, 3),
            "avg_duration_seconds": round(avg_duration, 2),
            "throughput_score": self._calculate_throughput_score(completion_rate, avg_duration)
        }

    def _calculate_throughput_score(self, completion_rate: float, avg_duration: float) -> str:
        """Calculate throughput quality score"""
        if completion_rate >= 0.8 and avg_duration < 300:
            return "excellent"
        elif completion_rate >= 0.6 and avg_duration < 600:
            return "good"
        elif completion_rate >= 0.4:
            return "fair"
        else:
            return "poor"

    def _analyze_consensus_quality(self, session: Dict) -> Dict:
        """Analyze consensus decision quality"""
        decisions = session.get('decisions', [])
        if not decisions:
            return {"status": "no_decisions"}

        confidences = [d.get('confidence', 0) for d in decisions]
        avg_confidence = statistics.mean(confidences) if confidences else 0

        # High confidence decisions (>= 0.75)
        high_confidence = sum(1 for c in confidences if c >= 0.75)
        low_confidence = sum(1 for c in confidences if c < 0.5)

        return {
            "total_decisions": len(decisions),
            "avg_confidence": round(avg_confidence, 3),
            "high_confidence_decisions": high_confidence,
            "low_confidence_decisions": low_confidence,
            "consensus_quality": self._rate_consensus_quality(avg_confidence, low_confidence, len(decisions))
        }

    def _rate_consensus_quality(self, avg_conf: float, low_conf: int, total: int) -> str:
        """Rate overall consensus quality"""
        low_conf_rate = low_conf / total if total > 0 else 0

        if avg_conf >= 0.8 and low_conf_rate < 0.1:
            return "excellent"
        elif avg_conf >= 0.7 and low_conf_rate < 0.2:
            return "good"
        elif avg_conf >= 0.6:
            return "fair"
        else:
            return "poor"

    def _analyze_memory_health(self) -> Dict:
        """Analyze collective memory health"""
        if not self.memory_db.exists():
            return {"status": "no_database"}

        conn = sqlite3.connect(str(self.memory_db))
        cursor = conn.cursor()

        # Count entries by type
        cursor.execute("SELECT type, COUNT(*) FROM memory GROUP BY type")
        type_counts = dict(cursor.fetchall())

        # Total entries
        cursor.execute("SELECT COUNT(*) FROM memory")
        total_entries = cursor.fetchone()[0]

        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM memory")
        avg_confidence = cursor.fetchone()[0] or 0

        # Recent activity (last hour)
        one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        cursor.execute("SELECT COUNT(*) FROM memory WHERE created_at > ?", (one_hour_ago,))
        recent_entries = cursor.fetchone()[0]

        # Association count
        cursor.execute("SELECT COUNT(*) FROM associations")
        association_count = cursor.fetchone()[0]

        conn.close()

        return {
            "total_entries": total_entries,
            "entries_by_type": type_counts,
            "avg_confidence": round(avg_confidence, 3),
            "recent_entries": recent_entries,
            "associations": association_count,
            "memory_health_score": self._rate_memory_health(total_entries, avg_confidence)
        }

    def _rate_memory_health(self, total: int, avg_conf: float) -> str:
        """Rate memory system health"""
        if total > 100 and avg_conf >= 0.8:
            return "excellent"
        elif total > 50 and avg_conf >= 0.7:
            return "good"
        elif total > 20:
            return "fair"
        else:
            return "poor"

    def _identify_bottlenecks(self, session: Dict) -> List[Dict]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        workers = session.get('workers', [])
        tasks = session.get('tasks', [])

        # Check worker availability
        if workers:
            idle_rate = sum(1 for w in workers if w.get('status') == 'idle') / len(workers)
            if idle_rate > 0.5:
                bottlenecks.append({
                    "type": "underutilization",
                    "severity": "medium",
                    "description": f"{idle_rate*100:.1f}% of workers are idle"
                })

        # Check task backlog
        if tasks:
            pending_rate = sum(1 for t in tasks if t.get('status') == 'pending') / len(tasks)
            if pending_rate > 0.3:
                bottlenecks.append({
                    "type": "task_backlog",
                    "severity": "high",
                    "description": f"{pending_rate*100:.1f}% of tasks are pending"
                })

        # Check for worker imbalance
        if workers:
            completion_counts = [w.get('tasks_completed', 0) for w in workers]
            if len(completion_counts) > 1:
                variance = statistics.variance(completion_counts)
                mean = statistics.mean(completion_counts)
                if mean > 0 and (variance ** 0.5) / mean > 0.5:
                    bottlenecks.append({
                        "type": "load_imbalance",
                        "severity": "medium",
                        "description": "Uneven task distribution across workers"
                    })

        return bottlenecks

    def _generate_recommendations(self, analysis: Dict):
        """Generate optimization recommendations"""
        self.recommendations = []

        # Worker utilization recommendations
        worker_util = analysis.get('worker_utilization', {})
        if worker_util.get('utilization_rate', 0) < 0.5:
            self.recommendations.append(OptimizationRecommendation(
                category="scaling",
                priority="medium",
                issue="Low worker utilization (<50%)",
                recommendation="Consider reducing max_workers or increasing task load",
                expected_improvement="30-40% resource savings"
            ))

        if worker_util.get('load_balance_score') in ['poor', 'fair']:
            self.recommendations.append(OptimizationRecommendation(
                category="load_balancing",
                priority="high",
                issue="Poor task distribution across workers",
                recommendation="Enable intelligent task assignment based on worker capabilities",
                expected_improvement="20-30% throughput increase"
            ))

        # Task efficiency recommendations
        task_eff = analysis.get('task_efficiency', {})
        if task_eff.get('completion_rate', 0) < 0.6:
            self.recommendations.append(OptimizationRecommendation(
                category="task_management",
                priority="high",
                issue="Low task completion rate (<60%)",
                recommendation="Review task timeouts and worker capacity allocation",
                expected_improvement="15-25% completion rate increase"
            ))

        # Consensus quality recommendations
        consensus = analysis.get('consensus_quality', {})
        if consensus.get('consensus_quality') in ['poor', 'fair']:
            self.recommendations.append(OptimizationRecommendation(
                category="consensus",
                priority="medium",
                issue="Low consensus confidence scores",
                recommendation="Consider Byzantine consensus algorithm for critical decisions",
                expected_improvement="40-50% confidence improvement"
            ))

        # Memory health recommendations
        memory = analysis.get('memory_health', {})
        if memory.get('total_entries', 0) > 10000:
            self.recommendations.append(OptimizationRecommendation(
                category="memory",
                priority="low",
                issue="Large memory footprint (>10k entries)",
                recommendation="Enable memory garbage collection and set TTL for temporary data",
                expected_improvement="50-60% memory reduction"
            ))

    def get_recommendations(self) -> List[Dict]:
        """Get optimization recommendations"""
        return [asdict(r) for r in self.recommendations]

    def export_report(self, analysis: Dict, filename: str = "optimization-report.json"):
        """Export comprehensive optimization report"""
        report = {
            "analysis": analysis,
            "recommendations": self.get_recommendations(),
            "generated_at": datetime.utcnow().isoformat()
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        return filename


def main():
    """Demo optimizer"""
    print("=== Hive Optimizer Demo ===\n")

    optimizer = HiveOptimizer()

    # Run performance analysis
    print("Running performance analysis...")
    analysis = optimizer.analyze_performance()

    print("\nPerformance Analysis:")
    print(json.dumps(analysis, indent=2))

    print("\n=== Optimization Recommendations ===")
    for rec in optimizer.get_recommendations():
        print(f"\n[{rec['priority'].upper()}] {rec['category']}")
        print(f"Issue: {rec['issue']}")
        print(f"Recommendation: {rec['recommendation']}")
        print(f"Expected Improvement: {rec['expected_improvement']}")

    # Export report
    report_file = optimizer.export_report(analysis)
    print(f"\nReport exported to: {report_file}")


if __name__ == "__main__":
    main()
