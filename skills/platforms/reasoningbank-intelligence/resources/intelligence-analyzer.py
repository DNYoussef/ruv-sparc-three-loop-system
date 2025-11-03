#!/usr/bin/env python3
"""
Intelligence Analyzer for ReasoningBank

Analyzes learning effectiveness, identifies improvement opportunities,
and provides meta-cognitive insights about agent performance.

Features:
- Learning curve analysis
- Strategy evolution tracking
- Performance regression detection
- Meta-learning insights
- Transfer learning opportunities
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LearningMetrics:
    """Metrics for learning effectiveness"""
    total_experiences: int
    success_rate: float
    improvement_rate: float  # Rate of improvement over time
    learning_efficiency: float  # Success rate / experiences
    strategy_diversity: float  # Number of strategies tried / total strategies
    adaptation_speed: float  # How quickly strategies are updated
    confidence_level: float  # Statistical confidence in metrics


class IntelligenceAnalyzer:
    """
    Meta-cognitive analyzer for ReasoningBank learning systems

    Analyzes:
    - Learning curves and trends
    - Strategy evolution
    - Performance regressions
    - Transfer learning potential
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.experiences = []
        self.strategies = {}
        self.learning_history = []

    def load_experiences(self, filepath: str) -> None:
        """Load experiences from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.experiences = data if isinstance(data, list) else [data]
        logger.info(f"Loaded {len(self.experiences)} experiences")

    def analyze_learning_curve(self) -> Dict[str, Any]:
        """
        Analyze learning curve over time

        Returns:
            Learning curve metrics and trend analysis
        """
        if len(self.experiences) < 10:
            return {'error': 'Insufficient data for learning curve analysis'}

        # Sort by timestamp
        sorted_exp = sorted(self.experiences, key=lambda e: e.get('timestamp', 0))

        # Calculate rolling success rate
        window_size = min(self.window_size, len(sorted_exp) // 4)
        success_rates = []
        timestamps = []

        for i in range(window_size, len(sorted_exp) + 1):
            window = sorted_exp[i-window_size:i]
            successes = sum(1 for e in window if e.get('outcome', {}).get('success', False))
            success_rate = successes / window_size
            success_rates.append(success_rate)
            timestamps.append(window[-1].get('timestamp', 0))

        # Fit linear regression to detect trend
        X = np.arange(len(success_rates)).reshape(-1, 1)
        y = np.array(success_rates)

        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0]
        r_squared = model.score(X, y)

        # Classify learning trend
        if slope > 0.001:
            trend = 'improving'
        elif slope < -0.001:
            trend = 'declining'
        else:
            trend = 'stable'

        return {
            'success_rates': success_rates,
            'timestamps': timestamps,
            'trend': trend,
            'improvement_rate': float(slope),
            'r_squared': float(r_squared),
            'initial_performance': float(success_rates[0]),
            'current_performance': float(success_rates[-1]),
            'improvement_delta': float(success_rates[-1] - success_rates[0])
        }

    def analyze_strategy_evolution(self) -> Dict[str, Any]:
        """
        Track how strategy usage evolves over time

        Returns:
            Strategy evolution metrics and insights
        """
        # Group experiences by time windows
        sorted_exp = sorted(self.experiences, key=lambda e: e.get('timestamp', 0))

        # Divide into time periods
        num_periods = 5
        period_size = len(sorted_exp) // num_periods

        strategy_evolution = []

        for i in range(num_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < num_periods - 1 else len(sorted_exp)
            period_exp = sorted_exp[start_idx:end_idx]

            # Count strategy usage
            strategy_counts = defaultdict(int)
            strategy_successes = defaultdict(int)

            for exp in period_exp:
                approach = exp.get('approach', 'unknown')
                strategy_counts[approach] += 1
                if exp.get('outcome', {}).get('success', False):
                    strategy_successes[approach] += 1

            # Calculate success rates
            period_stats = {
                'period': i + 1,
                'strategies': {}
            }

            for strategy, count in strategy_counts.items():
                success_rate = strategy_successes[strategy] / count if count > 0 else 0
                period_stats['strategies'][strategy] = {
                    'usage': count,
                    'success_rate': success_rate
                }

            strategy_evolution.append(period_stats)

        # Detect strategy shifts
        strategy_shifts = self._detect_strategy_shifts(strategy_evolution)

        return {
            'evolution': strategy_evolution,
            'strategy_shifts': strategy_shifts,
            'num_strategies_tried': len(set(e.get('approach', 'unknown') for e in sorted_exp)),
            'dominant_strategy': self._find_dominant_strategy(sorted_exp)
        }

    def _detect_strategy_shifts(self, evolution: List[Dict]) -> List[Dict]:
        """Detect significant changes in strategy usage"""
        shifts = []

        for i in range(1, len(evolution)):
            prev_period = evolution[i-1]['strategies']
            curr_period = evolution[i]['strategies']

            # Find strategies with significant usage changes
            for strategy in set(list(prev_period.keys()) + list(curr_period.keys())):
                prev_usage = prev_period.get(strategy, {}).get('usage', 0)
                curr_usage = curr_period.get(strategy, {}).get('usage', 0)

                total_prev = sum(s.get('usage', 0) for s in prev_period.values())
                total_curr = sum(s.get('usage', 0) for s in curr_period.values())

                prev_pct = prev_usage / total_prev if total_prev > 0 else 0
                curr_pct = curr_usage / total_curr if total_curr > 0 else 0

                if abs(curr_pct - prev_pct) > 0.2:  # 20% change threshold
                    shifts.append({
                        'period': i + 1,
                        'strategy': strategy,
                        'change': curr_pct - prev_pct,
                        'direction': 'increased' if curr_pct > prev_pct else 'decreased'
                    })

        return shifts

    def _find_dominant_strategy(self, experiences: List[Dict]) -> Dict[str, Any]:
        """Find most frequently used and successful strategy"""
        strategy_stats = defaultdict(lambda: {'count': 0, 'successes': 0})

        for exp in experiences:
            strategy = exp.get('approach', 'unknown')
            strategy_stats[strategy]['count'] += 1
            if exp.get('outcome', {}).get('success', False):
                strategy_stats[strategy]['successes'] += 1

        # Find strategy with best balance of usage and success
        best_strategy = None
        best_score = 0

        for strategy, stats in strategy_stats.items():
            if stats['count'] < 5:  # Minimum usage threshold
                continue

            success_rate = stats['successes'] / stats['count']
            # Score combines usage frequency and success rate
            score = (stats['count'] / len(experiences)) * success_rate

            if score > best_score:
                best_score = score
                best_strategy = strategy

        if best_strategy:
            return {
                'strategy': best_strategy,
                'usage_count': strategy_stats[best_strategy]['count'],
                'success_rate': strategy_stats[best_strategy]['successes'] / strategy_stats[best_strategy]['count'],
                'score': best_score
            }
        return {}

    def detect_performance_regressions(self) -> List[Dict[str, Any]]:
        """
        Detect periods of performance regression

        Returns:
            List of detected regressions with context
        """
        if len(self.experiences) < 20:
            return []

        # Calculate rolling performance
        sorted_exp = sorted(self.experiences, key=lambda e: e.get('timestamp', 0))
        window_size = min(20, len(sorted_exp) // 4)

        performance_windows = []

        for i in range(window_size, len(sorted_exp) + 1):
            window = sorted_exp[i-window_size:i]
            successes = sum(1 for e in window if e.get('outcome', {}).get('success', False))
            performance_windows.append({
                'end_index': i,
                'success_rate': successes / window_size,
                'timestamp': window[-1].get('timestamp', 0)
            })

        # Detect regressions (significant drops)
        regressions = []

        for i in range(1, len(performance_windows)):
            prev = performance_windows[i-1]['success_rate']
            curr = performance_windows[i]['success_rate']

            # Significant drop (>20% relative decrease)
            if curr < prev * 0.8 and prev > 0.5:
                # Statistical test
                window_start = sorted_exp[i*window_size-window_size:(i+1)*window_size-window_size]
                window_end = sorted_exp[i*window_size:(i+1)*window_size]

                successes_start = sum(1 for e in window_start if e.get('outcome', {}).get('success', False))
                successes_end = sum(1 for e in window_end if e.get('outcome', {}).get('success', False))

                # Two-proportion z-test
                contingency = [[successes_start, len(window_start) - successes_start],
                             [successes_end, len(window_end) - successes_end]]
                _, p_value = stats.fisher_exact(contingency)

                if p_value < 0.05:  # Statistically significant
                    regressions.append({
                        'period': i + 1,
                        'previous_rate': prev,
                        'current_rate': curr,
                        'drop_pct': (prev - curr) / prev * 100,
                        'p_value': p_value,
                        'timestamp': performance_windows[i]['timestamp']
                    })

        return regressions

    def identify_transfer_opportunities(self) -> List[Dict[str, Any]]:
        """
        Identify opportunities for transfer learning between similar tasks

        Returns:
            List of transfer learning opportunities
        """
        # Group by task type and context
        task_contexts = defaultdict(lambda: defaultdict(list))

        for exp in self.experiences:
            task_type = exp.get('task_type', 'unknown')
            context = json.dumps(exp.get('context', {}), sort_keys=True)
            task_contexts[task_type][context].append(exp)

        opportunities = []

        # Find task types with successful strategies
        for task_type, contexts in task_contexts.items():
            for context, exps in contexts.items():
                if len(exps) < 5:
                    continue

                # Find successful strategies
                strategy_success = defaultdict(lambda: {'count': 0, 'successes': 0})
                for exp in exps:
                    strategy = exp.get('approach', 'unknown')
                    strategy_success[strategy]['count'] += 1
                    if exp.get('outcome', {}).get('success', False):
                        strategy_success[strategy]['successes'] += 1

                for strategy, stats in strategy_success.items():
                    if stats['count'] >= 5:
                        success_rate = stats['successes'] / stats['count']

                        if success_rate > 0.7:  # High success rate
                            # Look for similar task types with low data
                            for other_task, other_contexts in task_contexts.items():
                                if other_task == task_type:
                                    continue

                                # Check if any context in other task has low data
                                for other_context, other_exps in other_contexts.items():
                                    if len(other_exps) < 5:
                                        # Context similarity (simple Jaccard)
                                        context_dict = json.loads(context)
                                        other_context_dict = json.loads(other_context)

                                        common_keys = set(context_dict.keys()) & set(other_context_dict.keys())
                                        all_keys = set(context_dict.keys()) | set(other_context_dict.keys())

                                        similarity = len(common_keys) / len(all_keys) if all_keys else 0

                                        if similarity > 0.5:
                                            opportunities.append({
                                                'from_task': task_type,
                                                'to_task': other_task,
                                                'strategy': strategy,
                                                'success_rate': success_rate,
                                                'context_similarity': similarity,
                                                'recommendation': f"Try {strategy} for {other_task} (works well for similar {task_type})"
                                            })

        return opportunities

    def get_learning_metrics(self) -> LearningMetrics:
        """Calculate comprehensive learning metrics"""
        if not self.experiences:
            return LearningMetrics(0, 0, 0, 0, 0, 0, 0)

        # Learning curve analysis
        curve = self.analyze_learning_curve()

        # Strategy diversity
        unique_strategies = len(set(e.get('approach', 'unknown') for e in self.experiences))
        strategy_diversity = unique_strategies / max(len(self.experiences), 1)

        # Adaptation speed (how quickly strategies are changed)
        strategy_changes = 0
        prev_strategy = None
        for exp in sorted(self.experiences, key=lambda e: e.get('timestamp', 0)):
            current_strategy = exp.get('approach', 'unknown')
            if prev_strategy and current_strategy != prev_strategy:
                strategy_changes += 1
            prev_strategy = current_strategy

        adaptation_speed = strategy_changes / len(self.experiences) if len(self.experiences) > 1 else 0

        # Overall success rate
        successes = sum(1 for e in self.experiences if e.get('outcome', {}).get('success', False))
        success_rate = successes / len(self.experiences)

        return LearningMetrics(
            total_experiences=len(self.experiences),
            success_rate=success_rate,
            improvement_rate=curve.get('improvement_rate', 0),
            learning_efficiency=success_rate / len(self.experiences),
            strategy_diversity=strategy_diversity,
            adaptation_speed=adaptation_speed,
            confidence_level=0.95 if len(self.experiences) > 100 else 0.8
        )

    def export_report(self, filepath: str) -> None:
        """Export comprehensive intelligence report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(self.get_learning_metrics()),
            'learning_curve': self.analyze_learning_curve(),
            'strategy_evolution': self.analyze_strategy_evolution(),
            'performance_regressions': self.detect_performance_regressions(),
            'transfer_opportunities': self.identify_transfer_opportunities()
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Intelligence report exported to {filepath}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: intelligence-analyzer.py <experiences.json>")
        sys.exit(1)

    analyzer = IntelligenceAnalyzer()
    analyzer.load_experiences(sys.argv[1])

    print("\n=== Learning Metrics ===")
    metrics = analyzer.get_learning_metrics()
    for field, value in asdict(metrics).items():
        print(f"{field}: {value}")

    print("\n=== Learning Curve ===")
    curve = analyzer.analyze_learning_curve()
    print(f"Trend: {curve.get('trend', 'unknown')}")
    print(f"Improvement Rate: {curve.get('improvement_rate', 0):.4f}")
    print(f"Initial → Current: {curve.get('initial_performance', 0):.2f} → {curve.get('current_performance', 0):.2f}")

    print("\n=== Strategy Evolution ===")
    evolution = analyzer.analyze_strategy_evolution()
    print(f"Strategies Tried: {evolution['num_strategies_tried']}")
    print(f"Dominant Strategy: {evolution.get('dominant_strategy', {}).get('strategy', 'none')}")

    print("\n=== Performance Regressions ===")
    regressions = analyzer.detect_performance_regressions()
    print(f"Detected: {len(regressions)} regression(s)")

    print("\n=== Transfer Learning Opportunities ===")
    opportunities = analyzer.identify_transfer_opportunities()
    for opp in opportunities[:3]:
        print(f"- {opp['recommendation']}")
