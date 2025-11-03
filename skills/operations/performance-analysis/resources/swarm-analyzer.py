#!/usr/bin/env python3
"""
Swarm Performance Analyzer
Comprehensive swarm metrics collection and analysis for Claude Flow swarms
"""

import json
import time
import statistics
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class AgentMetrics:
    """Individual agent performance metrics"""
    agent_id: str
    agent_type: str
    tasks_completed: int
    tasks_failed: int
    avg_task_time: float
    cpu_usage: float
    memory_usage: float
    message_count: int
    response_time: float
    utilization: float


@dataclass
class SwarmMetrics:
    """Aggregated swarm performance metrics"""
    swarm_id: str
    topology: str
    agent_count: int
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    avg_completion_time: float
    throughput: float
    efficiency_score: float
    bottleneck_agents: List[str]
    timestamp: str


class SwarmAnalyzer:
    """Analyze swarm performance and identify optimization opportunities"""

    def __init__(self, swarm_id: str, time_window: int = 3600):
        """
        Initialize analyzer

        Args:
            swarm_id: Swarm identifier
            time_window: Analysis time window in seconds (default 1 hour)
        """
        self.swarm_id = swarm_id
        self.time_window = time_window
        self.metrics_buffer = []
        self.agent_metrics = defaultdict(list)

    def collect_metrics(self, agent_data: Dict[str, Any]) -> None:
        """
        Collect real-time metrics from agents

        Args:
            agent_data: Raw agent metrics data
        """
        timestamp = datetime.now()

        # Store raw metrics with timestamp
        self.metrics_buffer.append({
            'timestamp': timestamp.isoformat(),
            'data': agent_data
        })

        # Update agent-specific metrics
        agent_id = agent_data.get('agent_id')
        if agent_id:
            self.agent_metrics[agent_id].append({
                'timestamp': timestamp,
                'metrics': agent_data
            })

        # Clean old metrics outside time window
        self._clean_old_metrics(timestamp)

    def _clean_old_metrics(self, current_time: datetime) -> None:
        """Remove metrics outside time window"""
        cutoff = current_time - timedelta(seconds=self.time_window)

        # Clean buffer
        self.metrics_buffer = [
            m for m in self.metrics_buffer
            if datetime.fromisoformat(m['timestamp']) > cutoff
        ]

        # Clean agent metrics
        for agent_id in list(self.agent_metrics.keys()):
            self.agent_metrics[agent_id] = [
                m for m in self.agent_metrics[agent_id]
                if m['timestamp'] > cutoff
            ]

            # Remove empty agent entries
            if not self.agent_metrics[agent_id]:
                del self.agent_metrics[agent_id]

    def analyze_agent_performance(self, agent_id: str) -> Optional[AgentMetrics]:
        """
        Analyze individual agent performance

        Args:
            agent_id: Agent identifier

        Returns:
            AgentMetrics object or None if no data
        """
        agent_data = self.agent_metrics.get(agent_id)
        if not agent_data:
            return None

        # Extract metrics
        task_times = []
        cpu_readings = []
        memory_readings = []
        message_counts = []
        response_times = []
        tasks_completed = 0
        tasks_failed = 0

        for entry in agent_data:
            metrics = entry['metrics']

            if 'task_time' in metrics:
                task_times.append(metrics['task_time'])
            if 'cpu_usage' in metrics:
                cpu_readings.append(metrics['cpu_usage'])
            if 'memory_usage' in metrics:
                memory_readings.append(metrics['memory_usage'])
            if 'message_count' in metrics:
                message_counts.append(metrics['message_count'])
            if 'response_time' in metrics:
                response_times.append(metrics['response_time'])

            tasks_completed += metrics.get('completed', 0)
            tasks_failed += metrics.get('failed', 0)

        # Calculate statistics
        avg_task_time = statistics.mean(task_times) if task_times else 0
        avg_cpu = statistics.mean(cpu_readings) if cpu_readings else 0
        avg_memory = statistics.mean(memory_readings) if memory_readings else 0
        avg_messages = statistics.mean(message_counts) if message_counts else 0
        avg_response = statistics.mean(response_times) if response_times else 0

        # Calculate utilization (tasks completed / total time active)
        total_time = sum(task_times) if task_times else 1
        active_time = len(agent_data) * 60  # Assuming 1-minute intervals
        utilization = min(total_time / active_time, 1.0) if active_time > 0 else 0

        return AgentMetrics(
            agent_id=agent_id,
            agent_type=agent_data[0]['metrics'].get('type', 'unknown'),
            tasks_completed=tasks_completed,
            tasks_failed=tasks_failed,
            avg_task_time=avg_task_time,
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            message_count=int(avg_messages),
            response_time=avg_response,
            utilization=utilization
        )

    def analyze_swarm_performance(self, topology: str) -> SwarmMetrics:
        """
        Analyze overall swarm performance

        Args:
            topology: Swarm topology type

        Returns:
            SwarmMetrics object
        """
        # Collect all agent metrics
        all_agents = []
        total_completed = 0
        total_failed = 0
        completion_times = []

        for agent_id in self.agent_metrics.keys():
            agent_metrics = self.analyze_agent_performance(agent_id)
            if agent_metrics:
                all_agents.append(agent_metrics)
                total_completed += agent_metrics.tasks_completed
                total_failed += agent_metrics.tasks_failed
                if agent_metrics.avg_task_time > 0:
                    completion_times.append(agent_metrics.avg_task_time)

        # Calculate swarm-level metrics
        agent_count = len(all_agents)
        total_tasks = total_completed + total_failed
        avg_completion = statistics.mean(completion_times) if completion_times else 0

        # Calculate throughput (tasks per minute)
        time_window_minutes = self.time_window / 60
        throughput = total_completed / time_window_minutes if time_window_minutes > 0 else 0

        # Calculate efficiency score (0-100)
        success_rate = total_completed / total_tasks if total_tasks > 0 else 0
        avg_utilization = statistics.mean([a.utilization for a in all_agents]) if all_agents else 0
        efficiency_score = (success_rate * 0.6 + avg_utilization * 0.4) * 100

        # Identify bottleneck agents (low utilization or high task times)
        bottlenecks = []
        if all_agents:
            avg_task_time = statistics.mean([a.avg_task_time for a in all_agents])
            for agent in all_agents:
                if agent.utilization < 0.5 or agent.avg_task_time > avg_task_time * 1.5:
                    bottlenecks.append(agent.agent_id)

        return SwarmMetrics(
            swarm_id=self.swarm_id,
            topology=topology,
            agent_count=agent_count,
            total_tasks=total_tasks,
            completed_tasks=total_completed,
            failed_tasks=total_failed,
            avg_completion_time=avg_completion,
            throughput=throughput,
            efficiency_score=efficiency_score,
            bottleneck_agents=bottlenecks,
            timestamp=datetime.now().isoformat()
        )

    def detect_bottlenecks(self, threshold: float = 0.2) -> Dict[str, List[Dict]]:
        """
        Detect performance bottlenecks

        Args:
            threshold: Bottleneck threshold (0-1), default 20%

        Returns:
            Dictionary of bottleneck categories with details
        """
        bottlenecks = {
            'communication': [],
            'processing': [],
            'memory': [],
            'coordination': []
        }

        # Analyze each agent
        for agent_id in self.agent_metrics.keys():
            agent_metrics = self.analyze_agent_performance(agent_id)
            if not agent_metrics:
                continue

            # Communication bottlenecks (high message count, slow response)
            if agent_metrics.response_time > 2.0:  # > 2 seconds
                bottlenecks['communication'].append({
                    'agent_id': agent_id,
                    'type': 'slow_response',
                    'value': agent_metrics.response_time,
                    'impact': min((agent_metrics.response_time / 2.0) * threshold, 1.0)
                })

            # Processing bottlenecks (high task times, low utilization)
            if agent_metrics.avg_task_time > 60:  # > 1 minute
                bottlenecks['processing'].append({
                    'agent_id': agent_id,
                    'type': 'slow_processing',
                    'value': agent_metrics.avg_task_time,
                    'impact': min((agent_metrics.avg_task_time / 60) * threshold, 1.0)
                })

            # Memory bottlenecks (high memory usage)
            if agent_metrics.memory_usage > 0.8:  # > 80%
                bottlenecks['memory'].append({
                    'agent_id': agent_id,
                    'type': 'high_memory',
                    'value': agent_metrics.memory_usage,
                    'impact': agent_metrics.memory_usage * threshold
                })

            # Coordination bottlenecks (low utilization)
            if agent_metrics.utilization < 0.5:  # < 50%
                bottlenecks['coordination'].append({
                    'agent_id': agent_id,
                    'type': 'low_utilization',
                    'value': agent_metrics.utilization,
                    'impact': (1 - agent_metrics.utilization) * threshold
                })

        return bottlenecks

    def generate_report(self, topology: str, format: str = 'json') -> str:
        """
        Generate performance analysis report

        Args:
            topology: Swarm topology type
            format: Output format ('json' or 'text')

        Returns:
            Formatted report string
        """
        swarm_metrics = self.analyze_swarm_performance(topology)
        bottlenecks = self.detect_bottlenecks()

        if format == 'json':
            return json.dumps({
                'swarm_metrics': asdict(swarm_metrics),
                'bottlenecks': bottlenecks
            }, indent=2)
        else:
            # Text format
            report = []
            report.append("=" * 60)
            report.append("SWARM PERFORMANCE ANALYSIS REPORT")
            report.append("=" * 60)
            report.append(f"\nSwarm ID: {swarm_metrics.swarm_id}")
            report.append(f"Topology: {swarm_metrics.topology}")
            report.append(f"Analysis Time: {swarm_metrics.timestamp}")
            report.append(f"\n{'-' * 60}")
            report.append("SUMMARY METRICS")
            report.append(f"{'-' * 60}")
            report.append(f"Agents: {swarm_metrics.agent_count}")
            report.append(f"Total Tasks: {swarm_metrics.total_tasks}")
            report.append(f"Completed: {swarm_metrics.completed_tasks}")
            report.append(f"Failed: {swarm_metrics.failed_tasks}")
            report.append(f"Avg Completion Time: {swarm_metrics.avg_completion_time:.2f}s")
            report.append(f"Throughput: {swarm_metrics.throughput:.2f} tasks/min")
            report.append(f"Efficiency Score: {swarm_metrics.efficiency_score:.1f}/100")

            # Bottlenecks section
            report.append(f"\n{'-' * 60}")
            report.append("BOTTLENECK ANALYSIS")
            report.append(f"{'-' * 60}")

            for category, issues in bottlenecks.items():
                if issues:
                    report.append(f"\n{category.upper()}:")
                    for issue in issues:
                        report.append(f"  - Agent {issue['agent_id']}: {issue['type']}")
                        report.append(f"    Value: {issue['value']:.2f}, Impact: {issue['impact']*100:.1f}%")

            if swarm_metrics.bottleneck_agents:
                report.append(f"\nBottleneck Agents: {', '.join(swarm_metrics.bottleneck_agents)}")

            report.append(f"\n{'=' * 60}")

            return '\n'.join(report)


def main():
    """Example usage"""
    analyzer = SwarmAnalyzer('swarm-123', time_window=3600)

    # Simulate metrics collection
    for i in range(10):
        analyzer.collect_metrics({
            'agent_id': f'agent-{i % 3}',
            'type': 'coder',
            'task_time': 45 + i * 5,
            'cpu_usage': 0.6 + (i % 3) * 0.1,
            'memory_usage': 0.5 + (i % 4) * 0.1,
            'message_count': 10 + i,
            'response_time': 1.5 + (i % 2) * 0.5,
            'completed': 1 if i % 5 != 0 else 0,
            'failed': 1 if i % 5 == 0 else 0
        })

    # Generate report
    print(analyzer.generate_report('mesh', format='text'))

    # Also save JSON
    with open('swarm-analysis.json', 'w') as f:
        f.write(analyzer.generate_report('mesh', format='json'))


if __name__ == '__main__':
    main()
