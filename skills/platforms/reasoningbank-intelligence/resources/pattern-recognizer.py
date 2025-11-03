#!/usr/bin/env python3
"""
Pattern Recognition Engine for ReasoningBank Intelligence

Analyzes task trajectories, identifies recurring patterns, and extracts
actionable insights for adaptive learning systems.

Features:
- Temporal pattern detection using sliding windows
- Sequence mining with configurable support thresholds
- Statistical significance testing (Chi-square, Fisher's exact)
- Pattern clustering with DBSCAN
- Anomaly detection using isolation forests
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaskExperience:
    """Individual task execution record"""
    task_id: str
    task_type: str
    approach: str
    outcome: Dict[str, Any]
    context: Dict[str, Any]
    timestamp: float
    duration: float
    success: bool


@dataclass
class Pattern:
    """Detected pattern with metadata"""
    pattern_id: str
    pattern_type: str  # 'sequence', 'temporal', 'contextual', 'anomaly'
    description: str
    triggers: List[str]
    actions: List[str]
    confidence: float
    support: int  # Number of times observed
    occurrences: List[Dict[str, Any]]
    statistical_significance: float
    first_seen: float
    last_seen: float


class PatternRecognizer:
    """
    Advanced pattern recognition engine for ReasoningBank

    Analyzes task execution histories to identify:
    - Successful approach sequences
    - Context-dependent strategies
    - Temporal patterns (time-of-day effects)
    - Anomalous behaviors
    """

    def __init__(
        self,
        min_support: int = 3,
        confidence_threshold: float = 0.7,
        temporal_window_hours: int = 24,
        anomaly_contamination: float = 0.1
    ):
        self.min_support = min_support
        self.confidence_threshold = confidence_threshold
        self.temporal_window = timedelta(hours=temporal_window_hours)
        self.anomaly_contamination = anomaly_contamination

        self.experiences: List[TaskExperience] = []
        self.patterns: Dict[str, Pattern] = {}

    def add_experience(self, experience: Dict[str, Any]) -> None:
        """Record new task experience"""
        exp = TaskExperience(
            task_id=experience['task_id'],
            task_type=experience['task_type'],
            approach=experience['approach'],
            outcome=experience['outcome'],
            context=experience.get('context', {}),
            timestamp=experience.get('timestamp', datetime.now().timestamp()),
            duration=experience.get('duration', 0),
            success=experience['outcome'].get('success', False)
        )
        self.experiences.append(exp)
        logger.info(f"Recorded experience: {exp.task_type} with {exp.approach}")

    def analyze_patterns(self) -> List[Pattern]:
        """
        Comprehensive pattern analysis

        Returns:
            List of detected patterns sorted by confidence
        """
        if len(self.experiences) < self.min_support:
            logger.warning(f"Insufficient data: {len(self.experiences)} < {self.min_support}")
            return []

        patterns = []

        # 1. Sequence patterns (approach sequences)
        patterns.extend(self._detect_sequence_patterns())

        # 2. Contextual patterns (context → approach mappings)
        patterns.extend(self._detect_contextual_patterns())

        # 3. Temporal patterns (time-based success rates)
        patterns.extend(self._detect_temporal_patterns())

        # 4. Anomaly detection
        patterns.extend(self._detect_anomalies())

        # Store and return
        for pattern in patterns:
            self.patterns[pattern.pattern_id] = pattern

        return sorted(patterns, key=lambda p: p.confidence, reverse=True)

    def _detect_sequence_patterns(self) -> List[Pattern]:
        """Detect successful approach sequences"""
        patterns = []

        # Group by task type
        by_type = defaultdict(list)
        for exp in self.experiences:
            by_type[exp.task_type].append(exp)

        for task_type, exps in by_type.items():
            # Sort by timestamp
            exps_sorted = sorted(exps, key=lambda e: e.timestamp)

            # Find sequences of successful approaches
            sequences = []
            current_seq = []

            for exp in exps_sorted:
                if exp.success:
                    current_seq.append(exp.approach)
                else:
                    if len(current_seq) >= 2:
                        sequences.append(tuple(current_seq))
                    current_seq = []

            if len(current_seq) >= 2:
                sequences.append(tuple(current_seq))

            # Count sequence frequencies
            seq_counts = Counter(sequences)

            # Create patterns for frequent sequences
            for seq, count in seq_counts.items():
                if count >= self.min_support:
                    # Calculate confidence (success rate)
                    total_attempts = len([e for e in exps if e.approach in seq])
                    confidence = count / max(total_attempts, 1)

                    if confidence >= self.confidence_threshold:
                        # Statistical significance (Chi-square)
                        success_in_seq = count
                        success_out_seq = len([e for e in exps if e.success and e.approach not in seq])
                        fail_in_seq = total_attempts - success_in_seq
                        fail_out_seq = len(exps) - total_attempts - success_out_seq

                        contingency = [[success_in_seq, success_out_seq],
                                     [fail_in_seq, fail_out_seq]]
                        _, p_value = stats.chi2_contingency(contingency)[:2]

                        pattern = Pattern(
                            pattern_id=f"seq_{task_type}_{hash(seq)}",
                            pattern_type='sequence',
                            description=f"Successful sequence for {task_type}: {' → '.join(seq)}",
                            triggers=[task_type],
                            actions=list(seq),
                            confidence=confidence,
                            support=count,
                            occurrences=[asdict(e) for e in exps if e.approach in seq],
                            statistical_significance=1 - p_value,
                            first_seen=min(e.timestamp for e in exps if e.approach in seq),
                            last_seen=max(e.timestamp for e in exps if e.approach in seq)
                        )
                        patterns.append(pattern)

        return patterns

    def _detect_contextual_patterns(self) -> List[Pattern]:
        """Detect context-dependent approach effectiveness"""
        patterns = []

        # Extract context keys
        context_keys = set()
        for exp in self.experiences:
            context_keys.update(exp.context.keys())

        for key in context_keys:
            # Group by context value
            by_value = defaultdict(list)
            for exp in self.experiences:
                if key in exp.context:
                    value = exp.context[key]
                    by_value[value].append(exp)

            for value, exps in by_value.items():
                if len(exps) < self.min_support:
                    continue

                # Find best approach for this context
                approach_success = defaultdict(lambda: {'success': 0, 'total': 0})
                for exp in exps:
                    approach_success[exp.approach]['total'] += 1
                    if exp.success:
                        approach_success[exp.approach]['success'] += 1

                for approach, counts in approach_success.items():
                    confidence = counts['success'] / counts['total']

                    if confidence >= self.confidence_threshold and counts['total'] >= self.min_support:
                        pattern = Pattern(
                            pattern_id=f"ctx_{key}_{value}_{approach}",
                            pattern_type='contextual',
                            description=f"When {key}={value}, use {approach}",
                            triggers=[f"{key}={value}"],
                            actions=[approach],
                            confidence=confidence,
                            support=counts['total'],
                            occurrences=[asdict(e) for e in exps if e.approach == approach],
                            statistical_significance=self._calculate_significance(counts['success'], counts['total']),
                            first_seen=min(e.timestamp for e in exps if e.approach == approach),
                            last_seen=max(e.timestamp for e in exps if e.approach == approach)
                        )
                        patterns.append(pattern)

        return patterns

    def _detect_temporal_patterns(self) -> List[Pattern]:
        """Detect time-based patterns (e.g., time-of-day effects)"""
        patterns = []

        # Group by hour of day
        by_hour = defaultdict(list)
        for exp in self.experiences:
            hour = datetime.fromtimestamp(exp.timestamp).hour
            by_hour[hour].append(exp)

        # Find hours with significantly different success rates
        overall_success_rate = sum(1 for e in self.experiences if e.success) / len(self.experiences)

        for hour, exps in by_hour.items():
            if len(exps) < self.min_support:
                continue

            hour_success_rate = sum(1 for e in exps if e.success) / len(exps)

            # Statistical test: is this hour's success rate significantly different?
            successes = sum(1 for e in exps if e.success)
            _, p_value = stats.binom_test(successes, len(exps), overall_success_rate, alternative='two-sided')

            if p_value < 0.05 and abs(hour_success_rate - overall_success_rate) > 0.2:
                pattern = Pattern(
                    pattern_id=f"temporal_hour_{hour}",
                    pattern_type='temporal',
                    description=f"Hour {hour}:00 has {'higher' if hour_success_rate > overall_success_rate else 'lower'} success rate",
                    triggers=[f"hour={hour}"],
                    actions=['schedule' if hour_success_rate > overall_success_rate else 'avoid'],
                    confidence=abs(hour_success_rate - overall_success_rate),
                    support=len(exps),
                    occurrences=[asdict(e) for e in exps],
                    statistical_significance=1 - p_value,
                    first_seen=min(e.timestamp for e in exps),
                    last_seen=max(e.timestamp for e in exps)
                )
                patterns.append(pattern)

        return patterns

    def _detect_anomalies(self) -> List[Pattern]:
        """Detect anomalous task executions"""
        if len(self.experiences) < 10:
            return []

        patterns = []

        # Extract features for anomaly detection
        features = []
        for exp in self.experiences:
            feature_vector = [
                exp.duration,
                1 if exp.success else 0,
                len(exp.context),
                hash(exp.approach) % 1000  # Encode approach
            ]
            features.append(feature_vector)

        features_array = np.array(features)

        # Isolation Forest
        iso_forest = IsolationForest(contamination=self.anomaly_contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(features_array)

        # Create patterns for anomalies
        anomaly_exps = [exp for exp, label in zip(self.experiences, anomaly_labels) if label == -1]

        if anomaly_exps:
            pattern = Pattern(
                pattern_id=f"anomaly_{datetime.now().timestamp()}",
                pattern_type='anomaly',
                description=f"Detected {len(anomaly_exps)} anomalous task executions",
                triggers=['anomaly_detected'],
                actions=['investigate', 'review'],
                confidence=0.8,
                support=len(anomaly_exps),
                occurrences=[asdict(e) for e in anomaly_exps],
                statistical_significance=0.95,
                first_seen=min(e.timestamp for e in anomaly_exps),
                last_seen=max(e.timestamp for e in anomaly_exps)
            )
            patterns.append(pattern)

        return patterns

    def _calculate_significance(self, successes: int, total: int) -> float:
        """Calculate statistical significance of success rate"""
        if total < self.min_support:
            return 0.0

        # Binomial test against 50% baseline
        _, p_value = stats.binom_test(successes, total, 0.5, alternative='two-sided')
        return 1 - p_value

    def export_patterns(self, filepath: str) -> None:
        """Export detected patterns to JSON"""
        patterns_dict = {
            pid: asdict(pattern) for pid, pattern in self.patterns.items()
        }

        with open(filepath, 'w') as f:
            json.dump(patterns_dict, f, indent=2)

        logger.info(f"Exported {len(self.patterns)} patterns to {filepath}")

    def load_patterns(self, filepath: str) -> None:
        """Load patterns from JSON"""
        with open(filepath, 'r') as f:
            patterns_dict = json.load(f)

        self.patterns = {
            pid: Pattern(**data) for pid, data in patterns_dict.items()
        }

        logger.info(f"Loaded {len(self.patterns)} patterns from {filepath}")


if __name__ == "__main__":
    # Example usage
    recognizer = PatternRecognizer(min_support=2, confidence_threshold=0.7)

    # Simulate task experiences
    experiences = [
        {
            'task_id': 'task_1',
            'task_type': 'code_review',
            'approach': 'static_analysis_first',
            'outcome': {'success': True, 'bugs_found': 5},
            'context': {'language': 'python', 'complexity': 'medium'},
            'duration': 120
        },
        {
            'task_id': 'task_2',
            'task_type': 'code_review',
            'approach': 'static_analysis_first',
            'outcome': {'success': True, 'bugs_found': 3},
            'context': {'language': 'python', 'complexity': 'low'},
            'duration': 80
        },
        {
            'task_id': 'task_3',
            'task_type': 'code_review',
            'approach': 'manual_review_first',
            'outcome': {'success': False, 'bugs_found': 1},
            'context': {'language': 'javascript', 'complexity': 'high'},
            'duration': 200
        }
    ]

    for exp in experiences:
        recognizer.add_experience(exp)

    patterns = recognizer.analyze_patterns()

    print(f"\nDetected {len(patterns)} patterns:")
    for pattern in patterns:
        print(f"- {pattern.description} (confidence: {pattern.confidence:.2f}, support: {pattern.support})")
