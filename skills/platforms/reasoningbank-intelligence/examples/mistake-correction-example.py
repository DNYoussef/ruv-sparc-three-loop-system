#!/usr/bin/env python3
"""
Mistake Correction Example for ReasoningBank Intelligence

Demonstrates how to learn from mistakes, identify failure patterns,
and implement corrective strategies. Shows complete workflow from
mistake detection to automated correction and prevention.

This example simulates a deployment system that learns from deployment
failures to improve success rates over time.
"""

import json
import sys
import os
from datetime import datetime, timedelta
import random
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'resources'))

from pattern_recognizer import PatternRecognizer
from intelligence_analyzer import IntelligenceAnalyzer, LearningMetrics


class DeploymentSystem:
    """Simulates a deployment system that can learn from failures"""

    def __init__(self):
        self.deployment_strategies = [
            'canary_deployment',
            'blue_green_deployment',
            'rolling_update',
            'recreate',
            'a_b_testing'
        ]

        # Common failure causes
        self.failure_causes = {
            'dependency_conflict': 0.15,
            'config_mismatch': 0.12,
            'resource_exhaustion': 0.10,
            'network_timeout': 0.08,
            'data_migration_failure': 0.06,
            'health_check_failure': 0.05
        }

        # Strategies that prevent specific failures
        self.preventive_strategies = {
            'dependency_conflict': ['blue_green_deployment', 'canary_deployment'],
            'config_mismatch': ['blue_green_deployment', 'a_b_testing'],
            'resource_exhaustion': ['rolling_update', 'canary_deployment'],
            'network_timeout': ['rolling_update', 'recreate'],
            'data_migration_failure': ['blue_green_deployment'],
            'health_check_failure': ['canary_deployment', 'a_b_testing']
        }

        self.mistake_history = []

    def simulate_deployment(self, strategy, context, learned_preventions=None):
        """Simulate a deployment with given strategy and context"""

        # Base success rate by strategy
        base_success_rates = {
            'canary_deployment': 0.85,
            'blue_green_deployment': 0.90,
            'rolling_update': 0.75,
            'recreate': 0.70,
            'a_b_testing': 0.80
        }

        success_prob = base_success_rates.get(strategy, 0.75)

        # Apply context modifiers
        if context.get('has_tests', False):
            success_prob *= 1.1
        if context.get('complexity') == 'high':
            success_prob *= 0.85
        if context.get('traffic') == 'high':
            success_prob *= 0.90

        # Check for preventable failures
        prevented_failures = []
        if learned_preventions:
            for failure_type, prevention_strategies in self.preventive_strategies.items():
                if failure_type in learned_preventions and strategy in prevention_strategies:
                    prevented_failures.append(failure_type)
                    success_prob += 0.05  # Boost for prevention

        # Determine outcome
        success = random.random() < min(success_prob, 0.98)

        outcome = {
            'success': success,
            'deployment_time': random.randint(300, 1800),
            'rollback_needed': not success
        }

        if not success:
            # Random failure cause (weighted)
            failure_cause = random.choices(
                list(self.failure_causes.keys()),
                weights=list(self.failure_causes.values())
            )[0]

            # Check if this failure could have been prevented
            preventable = strategy not in self.preventive_strategies.get(failure_cause, [])

            outcome['failure_cause'] = failure_cause
            outcome['preventable'] = preventable

            # Record mistake
            self.mistake_history.append({
                'timestamp': datetime.now().isoformat(),
                'strategy': strategy,
                'context': context,
                'failure_cause': failure_cause,
                'preventable': preventable
            })

        return outcome


class MistakeLearningSystem:
    """System for learning from mistakes and implementing corrections"""

    def __init__(self):
        self.pattern_recognizer = PatternRecognizer(min_support=2, confidence_threshold=0.6)
        self.intelligence_analyzer = IntelligenceAnalyzer(window_size=50)

        self.learned_corrections = defaultdict(list)
        self.prevention_effectiveness = defaultdict(lambda: {'tried': 0, 'prevented': 0})

    def analyze_failure(self, deployment_result, strategy, context):
        """Analyze a deployment failure and extract learnings"""

        if not deployment_result.get('failure_cause'):
            return None

        failure_pattern = {
            'failure_type': deployment_result['failure_cause'],
            'strategy_used': strategy,
            'context': context,
            'preventable': deployment_result.get('preventable', False)
        }

        # Record in pattern recognizer
        experience = {
            'task_id': f"deploy_{datetime.now().timestamp()}",
            'task_type': 'deployment',
            'approach': strategy,
            'outcome': deployment_result,
            'context': {
                **context,
                'failure_cause': deployment_result['failure_cause']
            },
            'timestamp': datetime.now().timestamp(),
            'duration': deployment_result.get('deployment_time', 0)
        }

        self.pattern_recognizer.add_experience(experience)

        return failure_pattern

    def identify_corrective_actions(self, failure_patterns):
        """Identify corrective actions from failure patterns"""

        # Analyze patterns
        detected_patterns = self.pattern_recognizer.analyze_patterns()

        corrections = []

        for pattern in detected_patterns:
            if pattern.pattern_type == 'contextual' and pattern.confidence > 0.7:
                # Extract failure cause from triggers
                failure_causes = [
                    t.split('=')[1] for t in pattern.triggers
                    if 'failure_cause=' in t
                ]

                if failure_causes:
                    correction = {
                        'failure_type': failure_causes[0],
                        'recommended_strategies': pattern.actions,
                        'confidence': pattern.confidence,
                        'evidence': pattern.support
                    }
                    corrections.append(correction)

        return corrections

    def apply_corrections(self, corrections):
        """Apply learned corrections to future deployments"""

        for correction in corrections:
            failure_type = correction['failure_type']
            strategies = correction['recommended_strategies']

            if failure_type not in self.learned_corrections:
                self.learned_corrections[failure_type] = []

            for strategy in strategies:
                if strategy not in self.learned_corrections[failure_type]:
                    self.learned_corrections[failure_type].append(strategy)

    def evaluate_learning_effectiveness(self, deployment_history):
        """Evaluate how well the system is learning from mistakes"""

        # Load history into analyzer
        for deployment in deployment_history:
            self.intelligence_analyzer.experiences.append(deployment)

        # Get learning metrics
        metrics = self.intelligence_analyzer.get_learning_metrics()

        # Analyze learning curve
        learning_curve = self.intelligence_analyzer.analyze_learning_curve()

        # Detect regressions
        regressions = self.intelligence_analyzer.detect_performance_regressions()

        return {
            'metrics': metrics,
            'learning_curve': learning_curve,
            'regressions': regressions
        }


def main():
    """Main execution flow"""

    print("=" * 80)
    print("ReasoningBank Intelligence - Mistake Correction Example")
    print("=" * 80)
    print()

    # Initialize systems
    deployment_system = DeploymentSystem()
    learning_system = MistakeLearningSystem()

    print("Step 1: Initial Deployment Phase (Learning from Mistakes)")
    print("-" * 80)

    deployment_history = []
    num_deployments = 150

    print(f"Running {num_deployments} deployments without corrections...\n")

    initial_failures = 0

    for i in range(num_deployments):
        # Random deployment configuration
        strategy = random.choice(deployment_system.deployment_strategies)
        context = {
            'environment': random.choice(['staging', 'production']),
            'has_tests': random.random() > 0.3,
            'complexity': random.choice(['low', 'medium', 'high']),
            'traffic': random.choice(['low', 'medium', 'high'])
        }

        # Simulate deployment
        result = deployment_system.simulate_deployment(strategy, context)

        deployment_record = {
            'task_id': f'deploy_{i}',
            'task_type': 'deployment',
            'approach': strategy,
            'outcome': result,
            'context': context,
            'timestamp': datetime.now().timestamp() + i * 600,
            'duration': result.get('deployment_time', 0)
        }

        deployment_history.append(deployment_record)

        # Analyze failures
        if not result['success']:
            initial_failures += 1
            learning_system.analyze_failure(result, strategy, context)

        if (i + 1) % 30 == 0:
            current_success_rate = (i + 1 - initial_failures) / (i + 1)
            print(f"  Deployments: {i+1}/{num_deployments}, "
                  f"Success Rate: {current_success_rate:.2%}, "
                  f"Failures: {initial_failures}")

    initial_success_rate = (num_deployments - initial_failures) / num_deployments

    print(f"\nInitial Phase Complete:")
    print(f"  Total Deployments: {num_deployments}")
    print(f"  Failures: {initial_failures}")
    print(f"  Success Rate: {initial_success_rate:.2%}")
    print()

    print("Step 2: Analyzing Failure Patterns")
    print("-" * 80)

    # Identify patterns
    patterns = learning_system.pattern_recognizer.analyze_patterns()
    print(f"Detected {len(patterns)} patterns from deployment history\n")

    # Group failures by cause
    failure_counts = defaultdict(int)
    for mistake in deployment_system.mistake_history:
        failure_counts[mistake['failure_cause']] += 1

    print("Failure Analysis:")
    for cause, count in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / initial_failures * 100) if initial_failures > 0 else 0
        print(f"  {cause:30s}: {count:3d} ({percentage:5.1f}%)")
    print()

    print("Step 3: Identifying Corrective Actions")
    print("-" * 80)

    corrections = learning_system.identify_corrective_actions(deployment_system.mistake_history)
    print(f"Identified {len(corrections)} corrective action(s):\n")

    for i, correction in enumerate(corrections, 1):
        print(f"  Correction {i}:")
        print(f"    Failure Type: {correction['failure_type']}")
        print(f"    Recommended: {', '.join(correction['recommended_strategies'])}")
        print(f"    Confidence: {correction['confidence']:.2%}")
        print(f"    Evidence: {correction['evidence']} occurrences")
        print()

    # Apply corrections
    learning_system.apply_corrections(corrections)
    learned_preventions = {
        failure_type: strategies[0] if strategies else None
        for failure_type, strategies in learning_system.learned_corrections.items()
    }

    print("Step 4: Applying Learned Corrections")
    print("-" * 80)

    print(f"Applied {len(learned_preventions)} correction strategies\n")

    # Second phase with corrections
    print("Running 150 more deployments WITH learned corrections...\n")

    corrected_failures = 0
    prevented_failures = 0

    for i in range(num_deployments):
        strategy = random.choice(deployment_system.deployment_strategies)
        context = {
            'environment': random.choice(['staging', 'production']),
            'has_tests': random.random() > 0.3,
            'complexity': random.choice(['low', 'medium', 'high']),
            'traffic': random.choice(['low', 'medium', 'high'])
        }

        # Simulate with learned preventions
        result = deployment_system.simulate_deployment(
            strategy,
            context,
            learned_preventions
        )

        deployment_record = {
            'task_id': f'deploy_corrected_{i}',
            'task_type': 'deployment',
            'approach': strategy,
            'outcome': result,
            'context': context,
            'timestamp': datetime.now().timestamp() + (num_deployments + i) * 600,
            'duration': result.get('deployment_time', 0)
        }

        deployment_history.append(deployment_record)

        if not result['success']:
            corrected_failures += 1
        elif result.get('prevented_failures'):
            prevented_failures += len(result['prevented_failures'])

        if (i + 1) % 30 == 0:
            current_success_rate = (i + 1 - corrected_failures) / (i + 1)
            print(f"  Deployments: {i+1}/{num_deployments}, "
                  f"Success Rate: {current_success_rate:.2%}, "
                  f"Failures: {corrected_failures}")

    corrected_success_rate = (num_deployments - corrected_failures) / num_deployments

    print(f"\nCorrected Phase Complete:")
    print(f"  Total Deployments: {num_deployments}")
    print(f"  Failures: {corrected_failures}")
    print(f"  Success Rate: {corrected_success_rate:.2%}")
    print()

    print("Step 5: Evaluating Improvement")
    print("-" * 80)

    improvement = corrected_success_rate - initial_success_rate
    improvement_pct = (improvement / initial_success_rate) * 100

    print(f"Performance Improvement:")
    print(f"  Before Corrections: {initial_success_rate:.2%}")
    print(f"  After Corrections:  {corrected_success_rate:.2%}")
    print(f"  Improvement:        {improvement:.2%} ({improvement_pct:+.1f}%)")
    print(f"  Failures Prevented: {initial_failures - corrected_failures}")
    print()

    print("Step 6: Learning Effectiveness Analysis")
    print("-" * 80)

    evaluation = learning_system.evaluate_learning_effectiveness(deployment_history)

    metrics = evaluation['metrics']
    print(f"Learning Metrics:")
    print(f"  Total Experiences: {metrics.total_experiences}")
    print(f"  Success Rate: {metrics.success_rate:.2%}")
    print(f"  Improvement Rate: {metrics.improvement_rate:.4f}")
    print(f"  Learning Efficiency: {metrics.learning_efficiency:.4f}")
    print(f"  Strategy Diversity: {metrics.strategy_diversity:.2%}")
    print(f"  Adaptation Speed: {metrics.adaptation_speed:.2%}")
    print(f"  Confidence Level: {metrics.confidence_level:.2%}")
    print()

    learning_curve = evaluation['learning_curve']
    if 'error' not in learning_curve:
        print(f"Learning Curve:")
        print(f"  Trend: {learning_curve['trend']}")
        print(f"  Improvement Rate: {learning_curve['improvement_rate']:.4f}")
        print(f"  Initial Performance: {learning_curve['initial_performance']:.2%}")
        print(f"  Current Performance: {learning_curve['current_performance']:.2%}")
        print(f"  Total Improvement: {learning_curve['improvement_delta']:.2%}")
        print()

    print("Step 7: Exporting Results")
    print("-" * 80)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    # Export deployment history
    history_file = os.path.join(output_dir, 'deployment_history.json')
    with open(history_file, 'w') as f:
        json.dump(deployment_history, f, indent=2)
    print(f"  Deployment history: {history_file}")

    # Export learning report
    report_file = os.path.join(output_dir, 'learning_report.json')
    learning_system.intelligence_analyzer.export_report(report_file)
    print(f"  Learning report: {report_file}")

    # Export mistake patterns
    mistakes_file = os.path.join(output_dir, 'mistake_patterns.json')
    with open(mistakes_file, 'w') as f:
        json.dump(deployment_system.mistake_history, f, indent=2)
    print(f"  Mistake patterns: {mistakes_file}")

    print()

    print("=" * 80)
    print("Mistake Correction Complete!")
    print("=" * 80)
    print()
    print("Key Insights:")
    print(f"  1. Learning from {initial_failures} failures improved success rate by {improvement_pct:.1f}%")
    print(f"  2. Pattern recognition identified {len(corrections)} corrective strategies")
    print(f"  3. Continuous learning enables ongoing improvement")
    print(f"  4. Mistake prevention is more effective than correction")
    print()
    print("Next Steps:")
    print("  1. Integrate with production deployment pipeline")
    print("  2. Implement automated rollback based on learned patterns")
    print("  3. Create alerting for preventable failures")
    print("  4. Share learned corrections across teams")
    print()


if __name__ == "__main__":
    main()
