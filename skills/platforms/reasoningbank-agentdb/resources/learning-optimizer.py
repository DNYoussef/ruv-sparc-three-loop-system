#!/usr/bin/env python3
"""
Learning Optimizer for ReasoningBank with AgentDB
Optimizes learning parameters and consolidates patterns for improved agent performance.
"""

import sqlite3
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
import argparse


class LearningOptimizer:
    """Optimize learning parameters and consolidate patterns."""

    def __init__(self, db_path: str, verbose: bool = False):
        self.db_path = Path(db_path)
        self.verbose = verbose
        self.stats = {
            'patterns_analyzed': 0,
            'patterns_consolidated': 0,
            'patterns_pruned': 0,
            'quality_improved': 0.0
        }

    def log(self, message: str, level: str = 'INFO'):
        """Log message with timestamp."""
        if self.verbose or level == 'ERROR':
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] [{level}] {message}")

    def load_patterns(self, domain: Optional[str] = None) -> List[Dict]:
        """Load patterns from AgentDB."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if domain:
            cursor.execute("""
                SELECT id, type, domain, pattern_data, confidence,
                       usage_count, success_count, created_at, last_used
                FROM patterns WHERE domain = ?
            """, (domain,))
        else:
            cursor.execute("""
                SELECT id, type, domain, pattern_data, confidence,
                       usage_count, success_count, created_at, last_used
                FROM patterns
            """)

        patterns = []
        for row in cursor.fetchall():
            pattern_data = json.loads(row[3])
            patterns.append({
                'id': row[0],
                'type': row[1],
                'domain': row[2],
                'pattern': pattern_data.get('pattern', {}),
                'embedding': pattern_data.get('embedding', []),
                'confidence': row[4],
                'usage_count': row[5],
                'success_count': row[6],
                'created_at': row[7],
                'last_used': row[8]
            })

        conn.close()
        self.stats['patterns_analyzed'] = len(patterns)
        self.log(f"Loaded {len(patterns)} patterns")
        return patterns

    def compute_pattern_quality(self, pattern: Dict) -> float:
        """Compute overall quality score for a pattern."""
        success_rate = (pattern['success_count'] / pattern['usage_count']
                       if pattern['usage_count'] > 0 else 0.5)

        # Usage score (logarithmic scaling)
        usage_score = min(np.log1p(pattern['usage_count']) / np.log1p(100), 1.0)

        # Confidence score
        confidence = pattern['confidence']

        # Recency score (patterns used recently are more valuable)
        age_days = (datetime.now().timestamp() * 1000 - pattern['last_used']) / (1000 * 86400)
        recency_score = np.exp(-age_days / 30)  # Decay over 30 days

        # Weighted combination
        quality = (0.4 * success_rate +
                  0.3 * confidence +
                  0.2 * usage_score +
                  0.1 * recency_score)

        return quality

    def identify_duplicates(self, patterns: List[Dict], similarity_threshold: float = 0.95) -> List[Tuple[str, str]]:
        """Identify duplicate or near-duplicate patterns."""
        duplicates = []

        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                p1, p2 = patterns[i], patterns[j]

                # Skip if different domains
                if p1['domain'] != p2['domain']:
                    continue

                # Compute embedding similarity
                if p1['embedding'] and p2['embedding']:
                    emb1 = np.array(p1['embedding'])
                    emb2 = np.array(p2['embedding'])

                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

                    if similarity >= similarity_threshold:
                        duplicates.append((p1['id'], p2['id']))

        self.log(f"Found {len(duplicates)} duplicate pairs")
        return duplicates

    def consolidate_patterns(self, patterns: List[Dict], duplicate_pairs: List[Tuple[str, str]]) -> Dict:
        """Consolidate duplicate patterns into higher-quality versions."""
        pattern_map = {p['id']: p for p in patterns}
        consolidations = []

        for id1, id2 in duplicate_pairs:
            p1 = pattern_map.get(id1)
            p2 = pattern_map.get(id2)

            if not p1 or not p2:
                continue

            # Determine which pattern to keep (higher quality)
            q1 = self.compute_pattern_quality(p1)
            q2 = self.compute_pattern_quality(p2)

            if q1 >= q2:
                keep, remove = p1, p2
            else:
                keep, remove = p2, p1

            # Merge statistics
            consolidated = {
                'id': keep['id'],
                'usage_count': keep['usage_count'] + remove['usage_count'],
                'success_count': keep['success_count'] + remove['success_count'],
                'confidence': max(keep['confidence'], remove['confidence'])
            }

            consolidations.append({
                'keep': keep['id'],
                'remove': remove['id'],
                'merged': consolidated
            })

        self.stats['patterns_consolidated'] = len(consolidations)
        self.log(f"Consolidated {len(consolidations)} patterns")
        return consolidations

    def prune_low_quality_patterns(self, patterns: List[Dict], quality_threshold: float = 0.2) -> List[str]:
        """Prune patterns with low quality scores."""
        to_prune = []

        for pattern in patterns:
            quality = self.compute_pattern_quality(pattern)

            # Prune if quality is low AND pattern has sufficient usage data
            if quality < quality_threshold and pattern['usage_count'] >= 5:
                to_prune.append(pattern['id'])

        self.stats['patterns_pruned'] = len(to_prune)
        self.log(f"Identified {len(to_prune)} patterns for pruning")
        return to_prune

    def optimize_confidence_scores(self, patterns: List[Dict]) -> Dict[str, float]:
        """Recalibrate confidence scores based on actual performance."""
        updates = {}

        for pattern in patterns:
            if pattern['usage_count'] == 0:
                continue

            # Compute empirical success rate
            empirical_success = pattern['success_count'] / pattern['usage_count']

            # Current confidence
            current_confidence = pattern['confidence']

            # Adjust confidence towards empirical success rate
            # Use exponential moving average with more weight on empirical data
            alpha = min(0.7, pattern['usage_count'] / 10)  # More data = more trust
            new_confidence = alpha * empirical_success + (1 - alpha) * current_confidence

            # Only update if significant change
            if abs(new_confidence - current_confidence) > 0.05:
                updates[pattern['id']] = new_confidence

        self.log(f"Updating confidence for {len(updates)} patterns")
        return updates

    def apply_optimizations(self, consolidations: List[Dict],
                          prune_ids: List[str],
                          confidence_updates: Dict[str, float]) -> bool:
        """Apply optimizations to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Apply consolidations
            for consolidation in consolidations:
                merged = consolidation['merged']

                cursor.execute("""
                    UPDATE patterns
                    SET usage_count = ?, success_count = ?, confidence = ?
                    WHERE id = ?
                """, (
                    merged['usage_count'],
                    merged['success_count'],
                    merged['confidence'],
                    merged['id']
                ))

                cursor.execute("DELETE FROM patterns WHERE id = ?",
                             (consolidation['remove'],))

            # Prune low-quality patterns
            for pattern_id in prune_ids:
                cursor.execute("DELETE FROM patterns WHERE id = ?", (pattern_id,))

            # Update confidence scores
            for pattern_id, new_confidence in confidence_updates.items():
                cursor.execute("""
                    UPDATE patterns SET confidence = ? WHERE id = ?
                """, (new_confidence, pattern_id))

            conn.commit()
            conn.close()

            self.log("Successfully applied optimizations")
            return True

        except sqlite3.Error as e:
            self.log(f"Error applying optimizations: {e}", 'ERROR')
            return False

    def optimize(self, domain: Optional[str] = None,
                similarity_threshold: float = 0.95,
                quality_threshold: float = 0.2) -> Dict:
        """Execute full optimization workflow."""
        self.log(f"Starting optimization for domain: {domain or 'all'}")

        # Load patterns
        patterns = self.load_patterns(domain)

        if len(patterns) == 0:
            return {'success': True, 'message': 'No patterns to optimize', 'stats': self.stats}

        # Compute initial quality
        initial_quality = np.mean([self.compute_pattern_quality(p) for p in patterns])

        # Identify duplicates
        duplicates = self.identify_duplicates(patterns, similarity_threshold)

        # Consolidate patterns
        consolidations = self.consolidate_patterns(patterns, duplicates)

        # Prune low-quality patterns
        prune_ids = self.prune_low_quality_patterns(patterns, quality_threshold)

        # Optimize confidence scores
        confidence_updates = self.optimize_confidence_scores(patterns)

        # Apply optimizations
        success = self.apply_optimizations(consolidations, prune_ids, confidence_updates)

        # Compute final quality
        final_patterns = self.load_patterns(domain)
        final_quality = np.mean([self.compute_pattern_quality(p) for p in final_patterns])

        self.stats['quality_improved'] = final_quality - initial_quality

        result = {
            'success': success,
            'stats': self.stats,
            'initial_quality': initial_quality,
            'final_quality': final_quality,
            'improvement': self.stats['quality_improved']
        }

        self.log(f"Optimization complete: {json.dumps(self.stats, indent=2)}")

        return result


def main():
    parser = argparse.ArgumentParser(description='Optimize ReasoningBank learning patterns')
    parser.add_argument('--db', required=True, help='AgentDB database path')
    parser.add_argument('--domain', help='Filter by domain')
    parser.add_argument('--similarity-threshold', type=float, default=0.95,
                       help='Similarity threshold for duplicate detection')
    parser.add_argument('--quality-threshold', type=float, default=0.2,
                       help='Quality threshold for pruning')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    optimizer = LearningOptimizer(args.db, args.verbose)
    result = optimizer.optimize(args.domain, args.similarity_threshold, args.quality_threshold)

    if result['success']:
        print(f"\n✓ Optimization successful!")
        print(f"  Patterns analyzed: {result['stats']['patterns_analyzed']}")
        print(f"  Patterns consolidated: {result['stats']['patterns_consolidated']}")
        print(f"  Patterns pruned: {result['stats']['patterns_pruned']}")
        print(f"  Quality improvement: {result['improvement']:.4f}")
        print(f"  Initial quality: {result['initial_quality']:.4f}")
        print(f"  Final quality: {result['final_quality']:.4f}")
        return 0
    else:
        print(f"\n✗ Optimization failed!")
        print(f"  Stats: {json.dumps(result['stats'], indent=2)}")
        return 1


if __name__ == '__main__':
    exit(main())
