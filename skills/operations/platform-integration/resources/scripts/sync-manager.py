#!/usr/bin/env python3
"""
Data Synchronization Manager
Bidirectional data sync with conflict resolution and state tracking
"""

import argparse
import json
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class SyncDirection(Enum):
    BIDIRECTIONAL = "bidirectional"
    SOURCE_TO_TARGET = "source_to_target"
    TARGET_TO_SOURCE = "target_to_source"


class ConflictStrategy(Enum):
    LAST_WRITE_WINS = "last_write_wins"
    SOURCE_WINS = "source_wins"
    TARGET_WINS = "target_wins"
    MERGE = "merge"
    MANUAL = "manual"


class SyncState:
    """Track synchronization state"""

    def __init__(self, state_file: str = "sync_state.json"):
        self.state_file = state_file
        self.state = self.load_state()

    def load_state(self) -> Dict[str, Any]:
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'last_sync': {},
                'checksums': {},
                'conflicts': []
            }

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def update_last_sync(self, entity: str, direction: str):
        if entity not in self.state['last_sync']:
            self.state['last_sync'][entity] = {}

        self.state['last_sync'][entity][direction] = datetime.now().isoformat()
        self.save_state()

    def get_last_sync(self, entity: str, direction: str) -> Optional[str]:
        return self.state['last_sync'].get(entity, {}).get(direction)

    def add_conflict(self, conflict: Dict[str, Any]):
        self.state['conflicts'].append({
            **conflict,
            'timestamp': datetime.now().isoformat()
        })
        self.save_state()


class DataSynchronizer:
    """Bidirectional data synchronization engine"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.direction = SyncDirection(config.get('direction', 'bidirectional'))
        self.conflict_strategy = ConflictStrategy(
            config.get('conflict_strategy', 'last_write_wins')
        )
        self.state = SyncState()
        self.batch_size = config.get('batch_size', 1000)

    def sync(self, source_connector, target_connector, entity: str):
        """Execute synchronization"""
        print(f"🔄 Synchronizing {entity} ({self.direction.value})")

        if self.direction in [SyncDirection.BIDIRECTIONAL, SyncDirection.SOURCE_TO_TARGET]:
            self._sync_direction(source_connector, target_connector, entity, 'source_to_target')

        if self.direction in [SyncDirection.BIDIRECTIONAL, SyncDirection.TARGET_TO_SOURCE]:
            self._sync_direction(target_connector, source_connector, entity, 'target_to_source')

        print(f"✅ Synchronization complete")

    def _sync_direction(self, source, target, entity: str, direction: str):
        """Sync in one direction"""
        print(f"  → {direction}")

        # Get last sync timestamp
        last_sync = self.state.get_last_sync(entity, direction)

        # Fetch changed records from source
        if last_sync:
            source_records = source.get_modified_since(entity, last_sync)
        else:
            source_records = source.get_all(entity)

        print(f"    Found {len(source_records)} records to sync")

        if not source_records:
            return

        # Process in batches
        for i in range(0, len(source_records), self.batch_size):
            batch = source_records[i:i + self.batch_size]
            self._sync_batch(batch, source, target, entity)

        # Update sync state
        self.state.update_last_sync(entity, direction)

    def _sync_batch(self, records: List[Dict[str, Any]], source, target, entity: str):
        """Sync a batch of records"""
        for record in records:
            try:
                # Check if record exists in target
                target_record = target.find_by_id(entity, record['id'])

                if target_record:
                    # Record exists - check for conflicts
                    if self._has_conflict(record, target_record):
                        resolved = self._resolve_conflict(record, target_record)
                        target.update(entity, record['id'], resolved)
                    else:
                        target.update(entity, record['id'], record)
                else:
                    # New record - create in target
                    target.create(entity, record)

            except Exception as e:
                print(f"    ⚠️  Error syncing record {record.get('id')}: {str(e)}")

    def _has_conflict(self, source_record: Dict[str, Any], target_record: Dict[str, Any]) -> bool:
        """Check if records conflict"""
        # Compare update timestamps
        source_updated = source_record.get('updated_at', source_record.get('modified_at'))
        target_updated = target_record.get('updated_at', target_record.get('modified_at'))

        if not source_updated or not target_updated:
            return False

        # Conflict if both modified since last sync
        return source_updated != target_updated

    def _resolve_conflict(self, source_record: Dict[str, Any], target_record: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicting records"""
        strategy = self.conflict_strategy

        if strategy == ConflictStrategy.LAST_WRITE_WINS:
            source_time = source_record.get('updated_at', '1970-01-01')
            target_time = target_record.get('updated_at', '1970-01-01')
            return source_record if source_time > target_time else target_record

        elif strategy == ConflictStrategy.SOURCE_WINS:
            return source_record

        elif strategy == ConflictStrategy.TARGET_WINS:
            return target_record

        elif strategy == ConflictStrategy.MERGE:
            # Merge non-conflicting fields
            merged = {**target_record}
            merge_fields = self.config.get('merge_fields', [])

            for field in merge_fields:
                if field in source_record:
                    merged[field] = source_record[field]

            return merged

        elif strategy == ConflictStrategy.MANUAL:
            # Store conflict for manual resolution
            self.state.add_conflict({
                'entity': 'unknown',  # Would need to pass entity
                'source': source_record,
                'target': target_record
            })
            return target_record  # Keep target for now

        return source_record


def generate_sync_engine(output_file: str, config: Dict[str, Any]):
    """Generate sync engine code"""

    code = f'''#!/usr/bin/env python3
"""
Auto-generated Synchronization Engine
Generated: {datetime.now().isoformat()}
"""

from sync_manager import DataSynchronizer, SyncDirection, ConflictStrategy


class CustomSyncEngine(DataSynchronizer):
    """Customized synchronization engine"""

    def __init__(self):
        config = {config}
        super().__init__(config)

    def transform_record(self, record, direction):
        """Transform record based on direction"""
        transformations = {transformations}

        if direction not in transformations:
            return record

        mapping = transformations[direction]
        transformed = {{}}

        for target_field, source_field in mapping.items():
            if source_field in record:
                transformed[target_field] = record[source_field]

        return transformed


if __name__ == '__main__':
    import sys
    from importlib import import_module

    # Import connectors
    source_module = sys.argv[1]  # e.g., "connectors.salesforce_connector"
    target_module = sys.argv[2]  # e.g., "connectors.hubspot_connector"
    entity = sys.argv[3]  # e.g., "contacts"

    source_class = import_module(source_module)
    target_class = import_module(target_module)

    # Initialize
    source = source_class.Connector()
    target = target_class.Connector()
    engine = CustomSyncEngine()

    # Run sync
    engine.sync(source, target, entity)
'''

    with open(output_file, 'w') as f:
        f.write(code)

    print(f"✅ Generated sync engine: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Sync Manager')
    parser.add_argument('--config', required=True, help='Configuration file')
    parser.add_argument('--output', required=True, help='Output file')
    parser.add_argument('--strategy', default='bidirectional',
                       choices=['bidirectional', 'source_to_target', 'target_to_source'])

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    config['direction'] = args.strategy

    generate_sync_engine(args.output, config)


if __name__ == '__main__':
    main()
