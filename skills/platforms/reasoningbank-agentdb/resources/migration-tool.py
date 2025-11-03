#!/usr/bin/env python3
"""
ReasoningBank Migration Tool
Migrates legacy ReasoningBank SQLite databases to AgentDB format with validation.
"""

import sqlite3
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

class ReasoningBankMigrator:
    """Migrate legacy ReasoningBank data to AgentDB format."""

    def __init__(self, source_db: str, target_db: str, verbose: bool = False):
        self.source_db = Path(source_db)
        self.target_db = Path(target_db)
        self.verbose = verbose
        self.stats = {
            'patterns_migrated': 0,
            'trajectories_migrated': 0,
            'experiences_migrated': 0,
            'errors': 0,
            'validation_failures': 0
        }

    def log(self, message: str, level: str = 'INFO'):
        """Log message with timestamp."""
        if self.verbose or level == 'ERROR':
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] [{level}] {message}")

    def validate_source_db(self) -> bool:
        """Validate source database structure."""
        self.log(f"Validating source database: {self.source_db}")

        if not self.source_db.exists():
            self.log(f"Source database not found: {self.source_db}", 'ERROR')
            return False

        try:
            conn = sqlite3.connect(self.source_db)
            cursor = conn.cursor()

            # Check for required tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            required_tables = ['patterns', 'trajectories', 'memories']
            missing_tables = [t for t in required_tables if t not in tables]

            if missing_tables:
                self.log(f"Missing required tables: {missing_tables}", 'ERROR')
                return False

            self.log("Source database validation successful")
            conn.close()
            return True

        except sqlite3.Error as e:
            self.log(f"Database validation error: {e}", 'ERROR')
            return False

    def initialize_target_db(self) -> bool:
        """Initialize AgentDB target database."""
        self.log(f"Initializing target database: {self.target_db}")

        try:
            conn = sqlite3.connect(self.target_db)
            cursor = conn.cursor()

            # Create AgentDB patterns table with enhanced schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    embedding BLOB,
                    confidence REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    last_used INTEGER NOT NULL,
                    metadata TEXT
                )
            """)

            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain ON patterns(domain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON patterns(type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON patterns(confidence)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON patterns(created_at)")

            # Migration metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS migration_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp INTEGER NOT NULL
                )
            """)

            conn.commit()
            conn.close()
            self.log("Target database initialized successfully")
            return True

        except sqlite3.Error as e:
            self.log(f"Target database initialization error: {e}", 'ERROR')
            return False

    def compute_embedding(self, text: str, dimension: int = 1536) -> np.ndarray:
        """
        Compute simple text embedding (placeholder for actual embedding model).
        In production, use OpenAI embeddings or similar.
        """
        # Simple hash-based embedding for migration (replace with real embeddings)
        import hashlib
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        # Expand to desired dimension
        embedding = np.zeros(dimension)
        for i in range(dimension):
            embedding[i] = hash_bytes[i % len(hash_bytes)] / 255.0

        return embedding

    def migrate_patterns(self) -> int:
        """Migrate pattern records from source to target."""
        self.log("Migrating patterns...")

        source_conn = sqlite3.connect(self.source_db)
        target_conn = sqlite3.connect(self.target_db)

        source_cursor = source_conn.cursor()
        target_cursor = target_conn.cursor()

        migrated = 0

        try:
            # Fetch all patterns
            source_cursor.execute("SELECT * FROM patterns")
            patterns = source_cursor.fetchall()

            for pattern in patterns:
                try:
                    # Legacy schema: id, domain, pattern_text, confidence, usage_count, success_count
                    pattern_id = pattern[0]
                    domain = pattern[1]
                    pattern_text = pattern[2]
                    confidence = float(pattern[3]) if pattern[3] else 0.0
                    usage_count = int(pattern[4]) if pattern[4] else 0
                    success_count = int(pattern[5]) if pattern[5] else 0

                    # Compute embedding
                    embedding = self.compute_embedding(pattern_text)
                    embedding_blob = embedding.tobytes()

                    # Create pattern data
                    pattern_data = json.dumps({
                        'pattern': pattern_text,
                        'embedding': embedding.tolist(),
                        'migrated_from': 'legacy_reasoningbank'
                    })

                    # Insert into target
                    target_cursor.execute("""
                        INSERT INTO patterns (
                            id, type, domain, pattern_data, embedding,
                            confidence, usage_count, success_count,
                            created_at, last_used
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pattern_id,
                        'pattern',
                        domain,
                        pattern_data,
                        embedding_blob,
                        confidence,
                        usage_count,
                        success_count,
                        int(datetime.now().timestamp() * 1000),
                        int(datetime.now().timestamp() * 1000)
                    ))

                    migrated += 1

                except Exception as e:
                    self.log(f"Error migrating pattern {pattern_id}: {e}", 'ERROR')
                    self.stats['errors'] += 1

            target_conn.commit()
            self.stats['patterns_migrated'] = migrated
            self.log(f"Migrated {migrated} patterns")

        finally:
            source_conn.close()
            target_conn.close()

        return migrated

    def migrate_trajectories(self) -> int:
        """Migrate trajectory records from source to target."""
        self.log("Migrating trajectories...")

        source_conn = sqlite3.connect(self.source_db)
        target_conn = sqlite3.connect(self.target_db)

        source_cursor = source_conn.cursor()
        target_cursor = target_conn.cursor()

        migrated = 0

        try:
            # Fetch all trajectories
            source_cursor.execute("SELECT * FROM trajectories")
            trajectories = source_cursor.fetchall()

            for trajectory in trajectories:
                try:
                    # Legacy schema: id, task, steps, outcome, metrics
                    traj_id = trajectory[0]
                    task = trajectory[1]
                    steps = trajectory[2]
                    outcome = trajectory[3]
                    metrics = trajectory[4]

                    # Create trajectory data
                    trajectory_data = {
                        'task': task,
                        'steps': json.loads(steps) if steps else [],
                        'outcome': outcome,
                        'metrics': json.loads(metrics) if metrics else {}
                    }

                    trajectory_text = json.dumps(trajectory_data)
                    embedding = self.compute_embedding(trajectory_text)
                    embedding_blob = embedding.tobytes()

                    pattern_data = json.dumps({
                        'embedding': embedding.tolist(),
                        'pattern': trajectory_data,
                        'migrated_from': 'legacy_reasoningbank'
                    })

                    # Determine success
                    success = 1 if outcome == 'success' else 0

                    target_cursor.execute("""
                        INSERT INTO patterns (
                            id, type, domain, pattern_data, embedding,
                            confidence, usage_count, success_count,
                            created_at, last_used
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        traj_id,
                        'trajectory',
                        task.split('-')[0] if '-' in task else 'general',
                        pattern_data,
                        embedding_blob,
                        0.8 if success else 0.3,
                        1,
                        success,
                        int(datetime.now().timestamp() * 1000),
                        int(datetime.now().timestamp() * 1000)
                    ))

                    migrated += 1

                except Exception as e:
                    self.log(f"Error migrating trajectory {traj_id}: {e}", 'ERROR')
                    self.stats['errors'] += 1

            target_conn.commit()
            self.stats['trajectories_migrated'] = migrated
            self.log(f"Migrated {migrated} trajectories")

        finally:
            source_conn.close()
            target_conn.close()

        return migrated

    def validate_migration(self) -> bool:
        """Validate migrated data integrity."""
        self.log("Validating migration...")

        try:
            conn = sqlite3.connect(self.target_db)
            cursor = conn.cursor()

            # Check record counts
            cursor.execute("SELECT COUNT(*) FROM patterns")
            total_patterns = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM patterns WHERE type='trajectory'")
            total_trajectories = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM patterns WHERE embedding IS NULL")
            missing_embeddings = cursor.fetchone()[0]

            if missing_embeddings > 0:
                self.log(f"Warning: {missing_embeddings} patterns missing embeddings", 'WARN')
                self.stats['validation_failures'] += missing_embeddings

            self.log(f"Validation complete: {total_patterns} patterns, {total_trajectories} trajectories")

            # Store migration metadata
            timestamp = int(datetime.now().timestamp() * 1000)
            cursor.execute("""
                INSERT OR REPLACE INTO migration_metadata (key, value, timestamp)
                VALUES (?, ?, ?)
            """, ('migration_stats', json.dumps(self.stats), timestamp))

            cursor.execute("""
                INSERT OR REPLACE INTO migration_metadata (key, value, timestamp)
                VALUES (?, ?, ?)
            """, ('source_db', str(self.source_db), timestamp))

            conn.commit()
            conn.close()

            return missing_embeddings == 0

        except sqlite3.Error as e:
            self.log(f"Validation error: {e}", 'ERROR')
            return False

    def migrate(self) -> Dict:
        """Execute full migration workflow."""
        self.log("Starting ReasoningBank migration")

        # Validate source
        if not self.validate_source_db():
            return {'success': False, 'error': 'Source validation failed', 'stats': self.stats}

        # Initialize target
        if not self.initialize_target_db():
            return {'success': False, 'error': 'Target initialization failed', 'stats': self.stats}

        # Migrate data
        self.migrate_patterns()
        self.migrate_trajectories()

        # Validate migration
        validation_success = self.validate_migration()

        result = {
            'success': validation_success,
            'stats': self.stats,
            'target_db': str(self.target_db)
        }

        self.log(f"Migration complete: {json.dumps(self.stats, indent=2)}")

        return result


def main():
    parser = argparse.ArgumentParser(description='Migrate ReasoningBank to AgentDB')
    parser.add_argument('--source', required=True, help='Source database path')
    parser.add_argument('--target', required=True, help='Target database path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    migrator = ReasoningBankMigrator(args.source, args.target, args.verbose)
    result = migrator.migrate()

    if result['success']:
        print(f"\n✓ Migration successful!")
        print(f"  Patterns: {result['stats']['patterns_migrated']}")
        print(f"  Trajectories: {result['stats']['trajectories_migrated']}")
        print(f"  Target: {result['target_db']}")
        sys.exit(0)
    else:
        print(f"\n✗ Migration failed!")
        print(f"  Error: {result.get('error', 'Unknown error')}")
        print(f"  Stats: {json.dumps(result['stats'], indent=2)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
