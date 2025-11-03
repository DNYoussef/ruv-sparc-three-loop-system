#!/usr/bin/env python3
"""
Session Memory Management with Triple-Layer Retention
Implements 24h/7d/30d+ retention policies for AgentDB
"""

import json
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class MemoryLayer:
    """Memory layer configuration"""
    name: str
    retention_hours: int
    max_entries: int
    priority_threshold: float


class TripleLayerMemory:
    """
    Triple-layer memory management system
    - Short-term: 24h retention, high-priority recent events
    - Mid-term: 7d retention, patterns and learnings
    - Long-term: 30d+ retention, core facts and knowledge
    """

    LAYERS = {
        'short_term': MemoryLayer('short_term', 24, 1000, 0.3),
        'mid_term': MemoryLayer('mid_term', 168, 5000, 0.6),  # 7 days
        'long_term': MemoryLayer('long_term', 720, 50000, 0.8),  # 30 days
    }

    def __init__(self, db_path: str = '.agentdb/memory.db'):
        """Initialize triple-layer memory system"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema"""
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS memory_layers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                layer TEXT NOT NULL,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                priority REAL DEFAULT 0.5,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed INTEGER,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_layer_session
                ON memory_layers(layer, session_id);
            CREATE INDEX IF NOT EXISTS idx_expires
                ON memory_layers(expires_at);
            CREATE INDEX IF NOT EXISTS idx_priority
                ON memory_layers(priority DESC);
        ''')
        self.conn.commit()

    def store(
        self,
        content: str,
        session_id: str,
        priority: float = 0.5,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Store memory entry with automatic layer assignment

        Args:
            content: Memory content
            session_id: Session identifier
            priority: Priority score (0.0-1.0)
            embedding: Vector embedding (optional)
            metadata: Additional metadata

        Returns:
            Memory entry ID
        """
        # Determine layer based on priority
        layer = self._assign_layer(priority)
        layer_config = self.LAYERS[layer]

        # Calculate expiration
        created_at = int(time.time())
        expires_at = created_at + (layer_config.retention_hours * 3600)

        # Serialize embedding
        embedding_blob = None
        if embedding is not None:
            embedding_blob = embedding.tobytes()

        # Insert entry
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO memory_layers
            (layer, session_id, content, embedding, priority,
             created_at, expires_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            layer,
            session_id,
            content,
            embedding_blob,
            priority,
            created_at,
            expires_at,
            json.dumps(metadata) if metadata else None
        ))
        self.conn.commit()

        # Enforce layer size limits
        self._enforce_limits(layer)

        return cursor.lastrowid

    def retrieve(
        self,
        session_id: str,
        layer: Optional[str] = None,
        limit: int = 20,
        min_priority: float = 0.0
    ) -> List[Dict]:
        """
        Retrieve memory entries

        Args:
            session_id: Session identifier
            layer: Specific layer (or all layers)
            limit: Maximum entries to return
            min_priority: Minimum priority threshold

        Returns:
            List of memory entries
        """
        now = int(time.time())

        query = '''
            SELECT id, layer, content, priority, created_at,
                   access_count, metadata
            FROM memory_layers
            WHERE session_id = ?
                AND expires_at > ?
                AND priority >= ?
        '''
        params = [session_id, now, min_priority]

        if layer:
            query += ' AND layer = ?'
            params.append(layer)

        query += ' ORDER BY priority DESC, created_at DESC LIMIT ?'
        params.append(limit)

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            entry = {
                'id': row[0],
                'layer': row[1],
                'content': row[2],
                'priority': row[3],
                'created_at': row[4],
                'access_count': row[5],
                'metadata': json.loads(row[6]) if row[6] else None
            }
            results.append(entry)

            # Update access statistics
            self._update_access(row[0])

        return results

    def consolidate(self, session_id: str) -> Dict[str, int]:
        """
        Consolidate memories across layers
        - Promote frequently accessed short-term to mid-term
        - Promote important mid-term to long-term

        Returns:
            Statistics of consolidation
        """
        stats = {'promoted': 0, 'expired': 0}
        now = int(time.time())

        # Promote short-term -> mid-term (high access, good priority)
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE memory_layers
            SET layer = 'mid_term',
                expires_at = ? + (168 * 3600),
                priority = priority * 1.1
            WHERE layer = 'short_term'
                AND session_id = ?
                AND access_count >= 5
                AND priority >= 0.5
                AND expires_at > ?
        ''', (now, session_id, now))
        stats['promoted'] += cursor.rowcount

        # Promote mid-term -> long-term (very high access, excellent priority)
        cursor.execute('''
            UPDATE memory_layers
            SET layer = 'long_term',
                expires_at = ? + (720 * 3600),
                priority = priority * 1.2
            WHERE layer = 'mid_term'
                AND session_id = ?
                AND access_count >= 10
                AND priority >= 0.7
                AND expires_at > ?
        ''', (now, session_id, now))
        stats['promoted'] += cursor.rowcount

        # Remove expired entries
        cursor.execute('''
            DELETE FROM memory_layers
            WHERE expires_at <= ?
        ''', (now,))
        stats['expired'] = cursor.rowcount

        self.conn.commit()
        return stats

    def get_statistics(self, session_id: str) -> Dict:
        """Get memory statistics for session"""
        cursor = self.conn.cursor()

        stats = {}
        for layer_name in self.LAYERS.keys():
            cursor.execute('''
                SELECT COUNT(*), AVG(priority), SUM(access_count)
                FROM memory_layers
                WHERE layer = ? AND session_id = ?
                    AND expires_at > ?
            ''', (layer_name, session_id, int(time.time())))

            row = cursor.fetchone()
            stats[layer_name] = {
                'count': row[0],
                'avg_priority': round(row[1], 3) if row[1] else 0,
                'total_accesses': row[2] or 0
            }

        return stats

    def _assign_layer(self, priority: float) -> str:
        """Assign layer based on priority"""
        if priority >= 0.8:
            return 'long_term'
        elif priority >= 0.5:
            return 'mid_term'
        else:
            return 'short_term'

    def _enforce_limits(self, layer: str):
        """Enforce size limits for layer"""
        layer_config = self.LAYERS[layer]
        now = int(time.time())

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM memory_layers
            WHERE layer = ? AND expires_at > ?
        ''', (layer, now))

        count = cursor.fetchone()[0]

        if count > layer_config.max_entries:
            # Remove lowest priority entries
            excess = count - layer_config.max_entries
            cursor.execute('''
                DELETE FROM memory_layers
                WHERE id IN (
                    SELECT id FROM memory_layers
                    WHERE layer = ? AND expires_at > ?
                    ORDER BY priority ASC, access_count ASC
                    LIMIT ?
                )
            ''', (layer, now, excess))
            self.conn.commit()

    def _update_access(self, entry_id: int):
        """Update access statistics"""
        self.conn.execute('''
            UPDATE memory_layers
            SET access_count = access_count + 1,
                last_accessed = ?
            WHERE id = ?
        ''', (int(time.time()), entry_id))
        self.conn.commit()

    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    """Example usage"""
    memory = TripleLayerMemory()

    # Store different priority memories
    session_id = "test-session-001"

    # High priority (long-term)
    memory.store(
        "User prefers Python for data analysis",
        session_id,
        priority=0.9,
        metadata={'category': 'preference', 'confidence': 'high'}
    )

    # Medium priority (mid-term)
    memory.store(
        "Discussion about machine learning algorithms",
        session_id,
        priority=0.6,
        metadata={'category': 'conversation', 'topic': 'ml'}
    )

    # Low priority (short-term)
    memory.store(
        "User asked about the weather",
        session_id,
        priority=0.3,
        metadata={'category': 'query', 'type': 'casual'}
    )

    # Retrieve all memories
    print("All memories:")
    memories = memory.retrieve(session_id, limit=10)
    for mem in memories:
        print(f"  [{mem['layer']}] {mem['content']} (priority: {mem['priority']})")

    # Get statistics
    print("\nMemory statistics:")
    stats = memory.get_statistics(session_id)
    for layer, data in stats.items():
        print(f"  {layer}: {data['count']} entries, "
              f"avg priority: {data['avg_priority']}")

    # Consolidate
    print("\nConsolidating memories...")
    consolidation_stats = memory.consolidate(session_id)
    print(f"  Promoted: {consolidation_stats['promoted']}")
    print(f"  Expired: {consolidation_stats['expired']}")

    memory.close()


if __name__ == '__main__':
    main()
