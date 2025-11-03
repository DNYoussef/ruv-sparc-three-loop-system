#!/usr/bin/env python3
"""
Workflow Optimization Example
Demonstrates analyzing and optimizing multi-stage workflows for performance

This example shows:
- Multi-stage workflow simulation
- Bottleneck identification in workflow stages
- Optimization recommendation generation
- Before/after performance comparison
"""

import time
import random
import statistics
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class WorkflowStage:
    """Represents a stage in the workflow"""
    name: str
    duration: float = 0
    agent_type: str = "generic"
    dependencies: List[str] = field(default_factory=list)
    parallel_capable: bool = False


@dataclass
class WorkflowMetrics:
    """Performance metrics for workflow execution"""
    total_duration: float
    stage_durations: Dict[str, float]
    bottleneck_stages: List[str]
    parallelization_factor: float
    efficiency_score: float


class WorkflowSimulator:
    """Simulates multi-stage workflow execution"""

    def __init__(self, name: str):
        self.name = name
        self.stages = []
        self.execution_history = []

    def add_stage(self, stage: WorkflowStage):
        """Add a stage to the workflow"""
        self.stages.append(stage)

    def execute(self, optimize: bool = False) -> WorkflowMetrics:
        """
        Execute the workflow

        Args:
            optimize: Whether to apply optimizations

        Returns:
            WorkflowMetrics with execution results
        """
        print(f"\n{'='*70}")
        print(f"Executing Workflow: {self.name}")
        print(f"Optimization: {'ENABLED' if optimize else 'DISABLED'}")
        print(f"{'='*70}\n")

        start_time = time.time()
        stage_durations = {}
        executed_stages = set()

        # Build dependency graph
        dependencies = {stage.name: stage.dependencies for stage in self.stages}

        # Execute stages
        while len(executed_stages) < len(self.stages):
            # Find executable stages (dependencies met)
            executable = [
                stage for stage in self.stages
                if stage.name not in executed_stages and
                all(dep in executed_stages for dep in stage.dependencies)
            ]

            if not executable:
                raise RuntimeError("Circular dependency detected!")

            # Group parallel-capable stages
            if optimize:
                parallel_stages = [s for s in executable if s.parallel_capable]
                sequential_stages = [s for s in executable if not s.parallel_capable]

                # Execute parallel stages concurrently (simulated)
                if parallel_stages:
                    print(f"Executing {len(parallel_stages)} stages in parallel:")
                    max_duration = 0
                    for stage in parallel_stages:
                        duration = self._execute_stage(stage, optimize)
                        stage_durations[stage.name] = duration
                        max_duration = max(max_duration, duration)
                        executed_stages.add(stage.name)
                        print(f"  ✓ {stage.name}: {duration:.2f}s")
                    time.sleep(max_duration)  # Simulate parallel execution

                # Execute sequential stages
                for stage in sequential_stages:
                    duration = self._execute_stage(stage, optimize)
                    stage_durations[stage.name] = duration
                    executed_stages.add(stage.name)
                    print(f"✓ {stage.name}: {duration:.2f}s")
                    time.sleep(duration)
            else:
                # Sequential execution
                for stage in executable:
                    duration = self._execute_stage(stage, optimize)
                    stage_durations[stage.name] = duration
                    executed_stages.add(stage.name)
                    print(f"✓ {stage.name}: {duration:.2f}s")
                    time.sleep(duration)

        total_duration = time.time() - start_time

        # Calculate metrics
        metrics = self._calculate_metrics(total_duration, stage_durations)

        self.execution_history.append({
            'timestamp': datetime.now(),
            'optimized': optimize,
            'metrics': metrics
        })

        return metrics

    def _execute_stage(self, stage: WorkflowStage, optimize: bool) -> float:
        """
        Simulate stage execution

        Args:
            stage: Stage to execute
            optimize: Whether optimizations are enabled

        Returns:
            Execution duration in seconds
        """
        base_duration = stage.duration

        # Add random variance (±15%)
        variance = random.uniform(-0.15, 0.15)
        duration = base_duration * (1 + variance)

        # Apply optimization benefits
        if optimize:
            if stage.parallel_capable:
                duration *= 0.7  # 30% improvement for parallel stages
            duration *= 0.9  # 10% general optimization

        return duration

    def _calculate_metrics(
        self,
        total_duration: float,
        stage_durations: Dict[str, float]
    ) -> WorkflowMetrics:
        """Calculate performance metrics"""

        # Identify bottlenecks (stages taking >20% of total time)
        avg_duration = statistics.mean(stage_durations.values())
        bottleneck_stages = [
            name for name, duration in stage_durations.items()
            if duration > avg_duration * 1.5
        ]

        # Calculate parallelization factor
        theoretical_sequential = sum(stage_durations.values())
        parallelization_factor = theoretical_sequential / total_duration if total_duration > 0 else 1.0

        # Calculate efficiency score (0-100)
        parallel_stages = len([s for s in self.stages if s.parallel_capable])
        total_stages = len(self.stages)
        parallel_ratio = parallel_stages / total_stages if total_stages > 0 else 0

        efficiency_score = min(
            (parallelization_factor / 2) * 50 +  # Up to 50 points for parallelization
            parallel_ratio * 30 +                 # Up to 30 points for parallel capability
            (1 - len(bottleneck_stages) / total_stages) * 20,  # Up to 20 points for no bottlenecks
            100
        )

        return WorkflowMetrics(
            total_duration=total_duration,
            stage_durations=stage_durations,
            bottleneck_stages=bottleneck_stages,
            parallelization_factor=parallelization_factor,
            efficiency_score=efficiency_score
        )


class WorkflowOptimizer:
    """Analyzes and optimizes workflows"""

    def __init__(self):
        self.recommendations = []

    def analyze(self, metrics: WorkflowMetrics) -> List[Dict]:
        """
        Analyze workflow and generate recommendations

        Args:
            metrics: WorkflowMetrics from execution

        Returns:
            List of optimization recommendations
        """
        self.recommendations = []

        # Analyze bottlenecks
        if metrics.bottleneck_stages:
            self.recommendations.append({
                'priority': 'high',
                'title': 'Optimize Bottleneck Stages',
                'description': f"Stages taking >50% longer than average: {', '.join(metrics.bottleneck_stages)}",
                'impact': '30-40% reduction in total workflow time',
                'actions': [
                    'Break down complex stages into smaller tasks',
                    'Spawn additional agents for bottleneck stages',
                    'Optimize algorithms in slow stages'
                ]
            })

        # Analyze parallelization
        if metrics.parallelization_factor < 1.5:
            self.recommendations.append({
                'priority': 'high',
                'title': 'Increase Parallelization',
                'description': f'Current parallelization factor: {metrics.parallelization_factor:.2f}x',
                'impact': '40-60% reduction in execution time',
                'actions': [
                    'Identify stages that can run concurrently',
                    'Reduce dependencies between stages',
                    'Use mesh topology for better parallel execution'
                ]
            })

        # Analyze efficiency
        if metrics.efficiency_score < 70:
            self.recommendations.append({
                'priority': 'medium',
                'title': 'Improve Overall Efficiency',
                'description': f'Efficiency score: {metrics.efficiency_score:.1f}/100',
                'impact': '20-30% improvement in resource utilization',
                'actions': [
                    'Optimize resource allocation',
                    'Implement caching for repeated operations',
                    'Enable agent auto-scaling'
                ]
            })

        # Stage-specific optimizations
        longest_stage = max(metrics.stage_durations.items(), key=lambda x: x[1])
        if longest_stage[1] > sum(metrics.stage_durations.values()) * 0.3:
            self.recommendations.append({
                'priority': 'critical',
                'title': f'Critical Bottleneck: {longest_stage[0]}',
                'description': f'Single stage consuming {longest_stage[1]/sum(metrics.stage_durations.values())*100:.1f}% of total time',
                'impact': 'Up to 50% reduction in workflow time',
                'actions': [
                    f'Decompose {longest_stage[0]} into parallel sub-tasks',
                    'Optimize core algorithms',
                    'Consider hardware acceleration'
                ]
            })

        return self.recommendations

    def generate_report(
        self,
        before_metrics: WorkflowMetrics,
        after_metrics: WorkflowMetrics = None
    ) -> str:
        """Generate optimization report"""

        lines = []
        lines.append('=' * 70)
        lines.append('WORKFLOW OPTIMIZATION ANALYSIS')
        lines.append('=' * 70)
        lines.append('')

        # Before metrics
        lines.append('BASELINE PERFORMANCE')
        lines.append('-' * 70)
        lines.append(f"Total Duration:          {before_metrics.total_duration:.2f}s")
        lines.append(f"Parallelization Factor:  {before_metrics.parallelization_factor:.2f}x")
        lines.append(f"Efficiency Score:        {before_metrics.efficiency_score:.1f}/100")
        lines.append(f"Bottleneck Stages:       {len(before_metrics.bottleneck_stages)}")
        lines.append('')

        # After metrics (if available)
        if after_metrics:
            lines.append('OPTIMIZED PERFORMANCE')
            lines.append('-' * 70)
            lines.append(f"Total Duration:          {after_metrics.total_duration:.2f}s")
            lines.append(f"Parallelization Factor:  {after_metrics.parallelization_factor:.2f}x")
            lines.append(f"Efficiency Score:        {after_metrics.efficiency_score:.1f}/100")
            lines.append(f"Bottleneck Stages:       {len(after_metrics.bottleneck_stages)}")
            lines.append('')

            # Improvements
            lines.append('IMPROVEMENTS')
            lines.append('-' * 70)
            time_saved = before_metrics.total_duration - after_metrics.total_duration
            time_improvement = (time_saved / before_metrics.total_duration) * 100
            lines.append(f"Time Saved:              {time_saved:.2f}s ({time_improvement:.1f}%)")

            parallel_improvement = ((after_metrics.parallelization_factor -
                                   before_metrics.parallelization_factor) /
                                  before_metrics.parallelization_factor) * 100
            lines.append(f"Parallelization:         +{parallel_improvement:.1f}%")

            efficiency_improvement = after_metrics.efficiency_score - before_metrics.efficiency_score
            lines.append(f"Efficiency:              +{efficiency_improvement:.1f} points")
            lines.append('')

        # Recommendations
        if self.recommendations:
            lines.append('OPTIMIZATION RECOMMENDATIONS')
            lines.append('-' * 70)
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. [{rec['priority'].upper()}] {rec['title']}")
                lines.append(f"   {rec['description']}")
                lines.append(f"   Impact: {rec['impact']}")
                lines.append(f"   Actions:")
                for action in rec['actions']:
                    lines.append(f"     - {action}")
                lines.append('')

        lines.append('=' * 70)

        return '\n'.join(lines)


def create_sample_workflow() -> WorkflowSimulator:
    """Create a sample multi-stage workflow"""

    workflow = WorkflowSimulator("Feature Development Pipeline")

    # Stage 1: Requirements Analysis
    workflow.add_stage(WorkflowStage(
        name="Requirements Analysis",
        duration=2.0,
        agent_type="researcher",
        dependencies=[],
        parallel_capable=False
    ))

    # Stage 2: Architecture Design
    workflow.add_stage(WorkflowStage(
        name="Architecture Design",
        duration=3.0,
        agent_type="architect",
        dependencies=["Requirements Analysis"],
        parallel_capable=False
    ))

    # Stage 3a: Backend Implementation (parallel)
    workflow.add_stage(WorkflowStage(
        name="Backend Implementation",
        duration=5.0,
        agent_type="coder",
        dependencies=["Architecture Design"],
        parallel_capable=True
    ))

    # Stage 3b: Frontend Implementation (parallel)
    workflow.add_stage(WorkflowStage(
        name="Frontend Implementation",
        duration=4.5,
        agent_type="coder",
        dependencies=["Architecture Design"],
        parallel_capable=True
    ))

    # Stage 3c: Database Schema (parallel)
    workflow.add_stage(WorkflowStage(
        name="Database Schema",
        duration=2.5,
        agent_type="coder",
        dependencies=["Architecture Design"],
        parallel_capable=True
    ))

    # Stage 4: Integration
    workflow.add_stage(WorkflowStage(
        name="Integration",
        duration=3.5,
        agent_type="coder",
        dependencies=[
            "Backend Implementation",
            "Frontend Implementation",
            "Database Schema"
        ],
        parallel_capable=False
    ))

    # Stage 5a: Unit Tests (parallel)
    workflow.add_stage(WorkflowStage(
        name="Unit Tests",
        duration=2.0,
        agent_type="tester",
        dependencies=["Integration"],
        parallel_capable=True
    ))

    # Stage 5b: Integration Tests (parallel)
    workflow.add_stage(WorkflowStage(
        name="Integration Tests",
        duration=3.0,
        agent_type="tester",
        dependencies=["Integration"],
        parallel_capable=True
    ))

    # Stage 6: Code Review
    workflow.add_stage(WorkflowStage(
        name="Code Review",
        duration=2.5,
        agent_type="reviewer",
        dependencies=["Unit Tests", "Integration Tests"],
        parallel_capable=False
    ))

    # Stage 7: Deployment
    workflow.add_stage(WorkflowStage(
        name="Deployment",
        duration=1.5,
        agent_type="devops",
        dependencies=["Code Review"],
        parallel_capable=False
    ))

    return workflow


def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("WORKFLOW OPTIMIZATION EXAMPLE")
    print("=" * 70 + "\n")

    # Create workflow
    workflow = create_sample_workflow()

    print("Workflow Stages:")
    for stage in workflow.stages:
        deps = f" (depends on: {', '.join(stage.dependencies)})" if stage.dependencies else ""
        parallel = " [PARALLEL]" if stage.parallel_capable else ""
        print(f"  - {stage.name}{parallel}{deps}")

    # Execute without optimization
    print("\n" + "=" * 70)
    print("BASELINE EXECUTION")
    print("=" * 70)
    before_metrics = workflow.execute(optimize=False)

    # Analyze and generate recommendations
    optimizer = WorkflowOptimizer()
    optimizer.analyze(before_metrics)

    # Execute with optimization
    print("\n" + "=" * 70)
    print("OPTIMIZED EXECUTION")
    print("=" * 70)
    after_metrics = workflow.execute(optimize=True)

    # Generate and display report
    report = optimizer.generate_report(before_metrics, after_metrics)
    print("\n" + report)


if __name__ == '__main__':
    main()
