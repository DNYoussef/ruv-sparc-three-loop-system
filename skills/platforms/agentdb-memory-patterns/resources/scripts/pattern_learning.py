#!/usr/bin/env python3
"""
Pattern Learning and Recognition for AgentDB
Implements pattern extraction, clustering, and recommendation
"""

import json
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import hashlib


@dataclass
class Pattern:
    """Learned pattern"""
    pattern_id: str
    trigger: str
    response: str
    confidence: float
    usage_count: int
    success_count: int
    context: Dict
    created_at: int
    last_used: int


class PatternLearner:
    """
    Pattern learning and recognition system
    Learns from successful interactions and applies patterns
    """

    def __init__(self, db_path: str = '.agentdb/patterns.db'):
        """Initialize pattern learner"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema"""
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                trigger TEXT NOT NULL,
                response TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                context TEXT,
                created_at INTEGER NOT NULL,
                last_used INTEGER,
                tags TEXT
            );

            CREATE TABLE IF NOT EXISTS pattern_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT NOT NULL,
                success INTEGER NOT NULL,
                execution_time INTEGER NOT NULL,
                context TEXT,
                feedback TEXT,
                FOREIGN KEY(pattern_id) REFERENCES patterns(pattern_id)
            );

            CREATE INDEX IF NOT EXISTS idx_trigger
                ON patterns(trigger);
            CREATE INDEX IF NOT EXISTS idx_confidence
                ON patterns(confidence DESC);
            CREATE INDEX IF NOT EXISTS idx_pattern_exec
                ON pattern_executions(pattern_id);
        ''')
        self.conn.commit()

    def learn_pattern(
        self,
        trigger: str,
        response: str,
        success: bool = True,
        context: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Learn a new pattern or update existing

        Args:
            trigger: Pattern trigger/input
            response: Pattern response/output
            success: Whether execution was successful
            context: Execution context
            tags: Pattern tags

        Returns:
            Pattern ID
        """
        # Generate pattern ID
        pattern_id = self._generate_pattern_id(trigger, response)

        cursor = self.conn.cursor()

        # Check if pattern exists
        cursor.execute(
            'SELECT pattern_id, usage_count, success_count FROM patterns WHERE pattern_id = ?',
            (pattern_id,)
        )
        existing = cursor.fetchone()

        current_time = int(time.time())

        if existing:
            # Update existing pattern
            usage_count = existing[1] + 1
            success_count = existing[2] + (1 if success else 0)
            confidence = success_count / usage_count

            cursor.execute('''
                UPDATE patterns
                SET usage_count = ?,
                    success_count = ?,
                    confidence = ?,
                    last_used = ?
                WHERE pattern_id = ?
            ''', (usage_count, success_count, confidence, current_time, pattern_id))
        else:
            # Create new pattern
            cursor.execute('''
                INSERT INTO patterns
                (pattern_id, trigger, response, confidence, usage_count,
                 success_count, context, created_at, last_used, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern_id,
                trigger,
                response,
                1.0 if success else 0.0,
                1,
                1 if success else 0,
                json.dumps(context) if context else None,
                current_time,
                current_time,
                json.dumps(tags) if tags else None
            ))

        # Log execution
        cursor.execute('''
            INSERT INTO pattern_executions
            (pattern_id, success, execution_time, context)
            VALUES (?, ?, ?, ?)
        ''', (
            pattern_id,
            1 if success else 0,
            current_time,
            json.dumps(context) if context else None
        ))

        self.conn.commit()
        return pattern_id

    def match_pattern(
        self,
        trigger: str,
        min_confidence: float = 0.5,
        context: Optional[Dict] = None
    ) -> Optional[Pattern]:
        """
        Match trigger to learned patterns

        Args:
            trigger: Trigger to match
            min_confidence: Minimum confidence threshold
            context: Current context for matching

        Returns:
            Best matching pattern or None
        """
        # Exact match first
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT pattern_id, trigger, response, confidence,
                   usage_count, success_count, context, created_at, last_used
            FROM patterns
            WHERE trigger = ? AND confidence >= ?
            ORDER BY confidence DESC, usage_count DESC
            LIMIT 1
        ''', (trigger, min_confidence))

        row = cursor.fetchone()
        if row:
            return self._row_to_pattern(row)

        # Fuzzy match (contains)
        cursor.execute('''
            SELECT pattern_id, trigger, response, confidence,
                   usage_count, success_count, context, created_at, last_used
            FROM patterns
            WHERE (trigger LIKE ? OR ? LIKE '%' || trigger || '%')
                AND confidence >= ?
            ORDER BY confidence DESC, usage_count DESC
            LIMIT 1
        ''', (f'%{trigger}%', trigger, min_confidence))

        row = cursor.fetchone()
        if row:
            return self._row_to_pattern(row)

        return None

    def get_top_patterns(
        self,
        limit: int = 10,
        min_usage: int = 1,
        tags: Optional[List[str]] = None
    ) -> List[Pattern]:
        """
        Get top patterns by confidence and usage

        Args:
            limit: Maximum patterns to return
            min_usage: Minimum usage count
            tags: Filter by tags

        Returns:
            List of top patterns
        """
        query = '''
            SELECT pattern_id, trigger, response, confidence,
                   usage_count, success_count, context, created_at, last_used
            FROM patterns
            WHERE usage_count >= ?
        '''
        params = [min_usage]

        if tags:
            # Simple tag matching (would need JSON support for complex queries)
            tag_conditions = ' OR '.join(['tags LIKE ?' for _ in tags])
            query += f' AND ({tag_conditions})'
            params.extend([f'%{tag}%' for tag in tags])

        query += ' ORDER BY confidence DESC, usage_count DESC LIMIT ?'
        params.append(limit)

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        return [self._row_to_pattern(row) for row in cursor.fetchall()]

    def analyze_pattern_performance(self, pattern_id: str) -> Dict:
        """
        Analyze pattern performance over time

        Returns:
            Performance statistics
        """
        cursor = self.conn.cursor()

        # Get pattern info
        cursor.execute('''
            SELECT trigger, response, confidence, usage_count, success_count
            FROM patterns WHERE pattern_id = ?
        ''', (pattern_id,))
        pattern_row = cursor.fetchone()

        if not pattern_row:
            return {}

        # Get execution history
        cursor.execute('''
            SELECT success, execution_time
            FROM pattern_executions
            WHERE pattern_id = ?
            ORDER BY execution_time DESC
            LIMIT 100
        ''', (pattern_id,))
        executions = cursor.fetchall()

        # Calculate statistics
        recent_success_rate = 0
        if executions:
            recent_successes = sum(1 for ex in executions[:10] if ex[0])
            recent_success_rate = recent_successes / min(10, len(executions))

        return {
            'pattern_id': pattern_id,
            'trigger': pattern_row[0],
            'response': pattern_row[1],
            'overall_confidence': pattern_row[2],
            'usage_count': pattern_row[3],
            'success_count': pattern_row[4],
            'recent_success_rate': recent_success_rate,
            'total_executions': len(executions)
        }

    def discover_patterns(
        self,
        session_logs: List[Dict],
        min_frequency: int = 3
    ) -> List[Tuple[str, str, int]]:
        """
        Discover patterns from session logs

        Args:
            session_logs: List of {trigger, response, success} dicts
            min_frequency: Minimum frequency to consider pattern

        Returns:
            List of (trigger, response, frequency) tuples
        """
        pattern_counts = Counter()

        for log in session_logs:
            if log.get('success', False):
                key = (log['trigger'], log['response'])
                pattern_counts[key] += 1

        # Filter by frequency
        discovered = [
            (trigger, response, count)
            for (trigger, response), count in pattern_counts.items()
            if count >= min_frequency
        ]

        # Sort by frequency
        discovered.sort(key=lambda x: x[2], reverse=True)

        return discovered

    def recommend_patterns(
        self,
        context: Dict,
        limit: int = 5
    ) -> List[Pattern]:
        """
        Recommend patterns based on context

        Args:
            context: Current context
            limit: Maximum recommendations

        Returns:
            List of recommended patterns
        """
        # Get all high-confidence patterns
        patterns = self.get_top_patterns(limit=50, min_usage=2)

        # Score patterns by context similarity
        scored_patterns = []
        for pattern in patterns:
            score = self._score_context_similarity(
                pattern.context,
                context
            )
            scored_patterns.append((pattern, score))

        # Sort by score
        scored_patterns.sort(key=lambda x: x[1], reverse=True)

        return [p for p, _ in scored_patterns[:limit]]

    def _generate_pattern_id(self, trigger: str, response: str) -> str:
        """Generate unique pattern ID"""
        content = f"{trigger}::{response}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _row_to_pattern(self, row: Tuple) -> Pattern:
        """Convert database row to Pattern"""
        return Pattern(
            pattern_id=row[0],
            trigger=row[1],
            response=row[2],
            confidence=row[3],
            usage_count=row[4],
            success_count=row[5],
            context=json.loads(row[6]) if row[6] else {},
            created_at=row[7],
            last_used=row[8]
        )

    def _score_context_similarity(
        self,
        pattern_context: Dict,
        current_context: Dict
    ) -> float:
        """Score context similarity (0-1)"""
        if not pattern_context or not current_context:
            return 0.0

        # Simple key overlap score
        pattern_keys = set(pattern_context.keys())
        current_keys = set(current_context.keys())

        if not pattern_keys or not current_keys:
            return 0.0

        overlap = len(pattern_keys & current_keys)
        total = len(pattern_keys | current_keys)

        return overlap / total if total > 0 else 0.0

    def close(self):
        """Close database connection"""
        self.conn.close()


import time

def main():
    """Example usage"""
    learner = PatternLearner()

    # Learn some patterns
    print("Learning patterns...")
    learner.learn_pattern(
        trigger="user_asks_time",
        response="provide_formatted_time",
        success=True,
        context={'timezone': 'UTC'},
        tags=['time', 'query']
    )

    learner.learn_pattern(
        trigger="user_greets",
        response="respond_greeting",
        success=True,
        context={'language': 'en'},
        tags=['greeting', 'social']
    )

    # Match patterns
    print("\nMatching patterns...")
    match = learner.match_pattern("user_asks_time")
    if match:
        print(f"Matched: {match.trigger} -> {match.response}")
        print(f"Confidence: {match.confidence}")

    # Get top patterns
    print("\nTop patterns:")
    top = learner.get_top_patterns(limit=5)
    for pattern in top:
        print(f"  {pattern.trigger}: {pattern.confidence:.2f} "
              f"({pattern.usage_count} uses)")

    # Analyze performance
    if top:
        print("\nPattern analysis:")
        analysis = learner.analyze_pattern_performance(top[0].pattern_id)
        for key, value in analysis.items():
            print(f"  {key}: {value}")

    learner.close()


if __name__ == '__main__':
    main()
