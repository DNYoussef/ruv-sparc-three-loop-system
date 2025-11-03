#!/usr/bin/env python3
"""
Pattern Recognition Example for ReasoningBank Intelligence

Demonstrates complete workflow for detecting patterns from task execution
histories, including sequence patterns, contextual patterns, temporal patterns,
and anomaly detection.

This example simulates a code review system that learns from past reviews
to identify successful review strategies.
"""

import json
import sys
import os
from datetime import datetime, timedelta
import random

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'resources'))

from pattern_recognizer import PatternRecognizer, Pattern


class CodeReviewSimulator:
    """Simulates a code review system with various strategies and contexts"""

    def __init__(self):
        self.strategies = [
            'static_analysis_first',
            'manual_review_first',
            'tdd_approach',
            'pair_review',
            'automated_only'
        ]

        self.languages = ['python', 'javascript', 'typescript', 'go', 'rust']
        self.complexities = ['low', 'medium', 'high']

        # Define strategy effectiveness by context
        self.strategy_effectiveness = {
            'static_analysis_first': {
                'python': 0.85,
                'javascript': 0.70,
                'typescript': 0.75,
                'go': 0.80,
                'rust': 0.90
            },
            'manual_review_first': {
                'python': 0.60,
                'javascript': 0.75,
                'typescript': 0.70,
                'go': 0.65,
                'rust': 0.60
            },
            'tdd_approach': {
                'python': 0.90,
                'javascript': 0.85,
                'typescript': 0.88,
                'go': 0.82,
                'rust': 0.85
            },
            'pair_review': {
                'python': 0.75,
                'javascript': 0.80,
                'typescript': 0.78,
                'go': 0.76,
                'rust': 0.74
            },
            'automated_only': {
                'python': 0.70,
                'javascript': 0.60,
                'typescript': 0.65,
                'go': 0.68,
                'rust': 0.75
            }
        }

    def simulate_review(self, task_id, strategy, language, complexity):
        """Simulate a code review with given parameters"""

        # Base success probability from strategy-language effectiveness
        base_prob = self.strategy_effectiveness[strategy][language]

        # Adjust for complexity
        complexity_modifiers = {'low': 1.1, 'medium': 1.0, 'high': 0.8}
        success_prob = base_prob * complexity_modifiers[complexity]

        # Add some randomness
        success_prob = min(success_prob + random.uniform(-0.1, 0.1), 1.0)

        success = random.random() < success_prob

        # Generate outcome metrics
        if success:
            bugs_found = random.randint(3, 10)
            false_positives = random.randint(0, 2)
            time_taken = random.randint(60, 180)
        else:
            bugs_found = random.randint(0, 2)
            false_positives = random.randint(2, 5)
            time_taken = random.randint(150, 300)

        return {
            'task_id': task_id,
            'task_type': 'code_review',
            'approach': strategy,
            'outcome': {
                'success': success,
                'bugs_found': bugs_found,
                'false_positives': false_positives
            },
            'context': {
                'language': language,
                'complexity': complexity
            },
            'timestamp': datetime.now().timestamp() + random.randint(0, 86400 * 30),
            'duration': time_taken
        }


def main():
    """Main execution flow"""

    print("=" * 80)
    print("ReasoningBank Intelligence - Pattern Recognition Example")
    print("=" * 80)
    print()

    # Initialize simulator and recognizer
    simulator = CodeReviewSimulator()
    recognizer = PatternRecognizer(
        min_support=3,
        confidence_threshold=0.7,
        temporal_window_hours=24
    )

    print("Step 1: Simulating Code Review Experiences")
    print("-" * 80)

    # Simulate 200 code reviews
    num_reviews = 200
    experiences = []

    for i in range(num_reviews):
        # Random selection of parameters
        strategy = random.choice(simulator.strategies)
        language = random.choice(simulator.languages)
        complexity = random.choice(simulator.complexities)

        # Weight towards certain patterns for demonstration
        if i % 10 == 0:  # Every 10th review, use Python + static_analysis (high success)
            language = 'python'
            strategy = 'static_analysis_first'
        elif i % 15 == 0:  # Every 15th review, use TDD approach (high success)
            strategy = 'tdd_approach'

        experience = simulator.simulate_review(
            task_id=f'review_{i}',
            strategy=strategy,
            language=language,
            complexity=complexity
        )

        experiences.append(experience)
        recognizer.add_experience(experience)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{num_reviews} reviews...")

    print(f"  Total reviews processed: {num_reviews}")
    print()

    # Calculate overall statistics
    total_success = sum(1 for e in experiences if e['outcome']['success'])
    overall_success_rate = total_success / len(experiences)

    print(f"  Overall success rate: {overall_success_rate:.2%}")
    print(f"  Total strategies used: {len(set(e['approach'] for e in experiences))}")
    print(f"  Total languages covered: {len(set(e['context']['language'] for e in experiences))}")
    print()

    print("Step 2: Analyzing Patterns")
    print("-" * 80)

    patterns = recognizer.analyze_patterns()

    print(f"  Detected {len(patterns)} patterns")
    print()

    # Group patterns by type
    patterns_by_type = {}
    for pattern in patterns:
        if pattern.pattern_type not in patterns_by_type:
            patterns_by_type[pattern.pattern_type] = []
        patterns_by_type[pattern.pattern_type].append(pattern)

    print("  Patterns by type:")
    for ptype, plist in patterns_by_type.items():
        print(f"    {ptype}: {len(plist)} pattern(s)")
    print()

    print("Step 3: Analyzing Sequence Patterns")
    print("-" * 80)

    sequence_patterns = patterns_by_type.get('sequence', [])
    if sequence_patterns:
        print(f"  Found {len(sequence_patterns)} sequence pattern(s):")
        print()

        for i, pattern in enumerate(sequence_patterns[:3], 1):
            print(f"  Pattern {i}:")
            print(f"    Description: {pattern.description}")
            print(f"    Confidence: {pattern.confidence:.2%}")
            print(f"    Support: {pattern.support} occurrences")
            print(f"    Actions: {' → '.join(pattern.actions)}")
            print(f"    Statistical Significance: {pattern.statistical_significance:.3f}")
            print()
    else:
        print("  No sequence patterns detected (may need more data or pattern clarity)")
        print()

    print("Step 4: Analyzing Contextual Patterns")
    print("-" * 80)

    contextual_patterns = patterns_by_type.get('contextual', [])
    if contextual_patterns:
        print(f"  Found {len(contextual_patterns)} contextual pattern(s):")
        print()

        # Sort by confidence
        contextual_patterns_sorted = sorted(
            contextual_patterns,
            key=lambda p: p.confidence,
            reverse=True
        )

        for i, pattern in enumerate(contextual_patterns_sorted[:5], 1):
            print(f"  Pattern {i}:")
            print(f"    Description: {pattern.description}")
            print(f"    Confidence: {pattern.confidence:.2%}")
            print(f"    Support: {pattern.support} occurrences")
            print(f"    Triggers: {', '.join(pattern.triggers)}")
            print(f"    Recommended Actions: {', '.join(pattern.actions)}")
            print()
    else:
        print("  No contextual patterns detected")
        print()

    print("Step 5: Analyzing Temporal Patterns")
    print("-" * 80)

    temporal_patterns = patterns_by_type.get('temporal', [])
    if temporal_patterns:
        print(f"  Found {len(temporal_patterns)} temporal pattern(s):")
        print()

        for i, pattern in enumerate(temporal_patterns, 1):
            print(f"  Pattern {i}:")
            print(f"    Description: {pattern.description}")
            print(f"    Confidence: {pattern.confidence:.2%}")
            print(f"    Support: {pattern.support} occurrences")
            print(f"    Statistical Significance: {pattern.statistical_significance:.3f}")
            print()
    else:
        print("  No temporal patterns detected (may need time-distributed data)")
        print()

    print("Step 6: Detecting Anomalies")
    print("-" * 80)

    anomaly_patterns = patterns_by_type.get('anomaly', [])
    if anomaly_patterns:
        print(f"  Found {len(anomaly_patterns)} anomaly pattern(s):")
        print()

        for i, pattern in enumerate(anomaly_patterns, 1):
            print(f"  Anomaly {i}:")
            print(f"    Description: {pattern.description}")
            print(f"    Support: {pattern.support} anomalous instances")
            print()

            # Show sample anomalies
            if pattern.occurrences:
                print(f"    Sample anomalous reviews:")
                for j, occurrence in enumerate(pattern.occurrences[:3], 1):
                    print(f"      {j}. Task {occurrence.get('task_id', 'unknown')}: "
                          f"Duration={occurrence.get('duration', 0)}s, "
                          f"Success={occurrence.get('outcome', {}).get('success', False)}")
                print()
    else:
        print("  No anomalies detected")
        print()

    print("Step 7: Generating Recommendations")
    print("-" * 80)

    # Extract actionable recommendations from high-confidence patterns
    recommendations = []

    for pattern in patterns:
        if pattern.confidence >= 0.8 and pattern.support >= 5:
            recommendations.append({
                'pattern_type': pattern.pattern_type,
                'description': pattern.description,
                'confidence': pattern.confidence,
                'actions': pattern.actions,
                'triggers': pattern.triggers
            })

    if recommendations:
        print(f"  Generated {len(recommendations)} high-confidence recommendation(s):")
        print()

        for i, rec in enumerate(sorted(recommendations, key=lambda r: r['confidence'], reverse=True)[:5], 1):
            print(f"  Recommendation {i}:")
            print(f"    {rec['description']}")
            print(f"    Confidence: {rec['confidence']:.2%}")
            print(f"    When: {', '.join(rec['triggers'])}")
            print(f"    Action: {', '.join(rec['actions'])}")
            print()
    else:
        print("  No high-confidence recommendations available yet")
        print()

    print("Step 8: Exporting Patterns")
    print("-" * 80)

    # Export patterns to file
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    patterns_file = os.path.join(output_dir, 'detected_patterns.json')
    recognizer.export_patterns(patterns_file)

    print(f"  Patterns exported to: {patterns_file}")
    print()

    # Export experiences for reference
    experiences_file = os.path.join(output_dir, 'code_review_experiences.json')
    with open(experiences_file, 'w') as f:
        json.dump(experiences, f, indent=2)

    print(f"  Experiences exported to: {experiences_file}")
    print()

    print("Step 9: Summary Statistics")
    print("-" * 80)

    print(f"  Total Patterns Detected: {len(patterns)}")
    print(f"  High-Confidence Patterns (≥80%): {len([p for p in patterns if p.confidence >= 0.8])}")
    print(f"  Well-Supported Patterns (≥10 occurrences): {len([p for p in patterns if p.support >= 10])}")
    print()

    # Strategy performance summary
    strategy_stats = {}
    for exp in experiences:
        strategy = exp['approach']
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {'total': 0, 'success': 0}
        strategy_stats[strategy]['total'] += 1
        if exp['outcome']['success']:
            strategy_stats[strategy]['success'] += 1

    print("  Strategy Performance Summary:")
    for strategy, stats in sorted(strategy_stats.items(), key=lambda x: x[1]['success']/x[1]['total'], reverse=True):
        success_rate = stats['success'] / stats['total']
        print(f"    {strategy:30s}: {success_rate:.2%} ({stats['success']}/{stats['total']})")
    print()

    print("=" * 80)
    print("Pattern Recognition Complete!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Review detected patterns in:", patterns_file)
    print("  2. Use high-confidence patterns for strategy recommendations")
    print("  3. Integrate with AdaptiveLearner for dynamic strategy selection")
    print("  4. Monitor pattern stability over time")
    print()


if __name__ == "__main__":
    main()
