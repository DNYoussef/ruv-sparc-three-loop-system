# Test 2: Long-Term Storage and Persistence

## Overview
Test long-term memory persistence, archival, and cross-session continuity.

**Status**: ✅ Ready for execution
**Estimated Duration**: 10-15 minutes
**Prerequisites**: AgentDB initialized, session_memory.py, test database

---

## Test Objectives

1. Verify long-term memory persistence across sessions
2. Test data integrity after database restarts
3. Validate archival and backup mechanisms
4. Confirm cross-session context retrieval
5. Test memory migration and versioning

---

## Test Setup

### 1. Initialize Test Environment

```bash
# Create persistent test database
export TEST_DB=".agentdb/test-longterm.db"
npx agentdb@latest init "$TEST_DB" --dimension 384

# Create multiple session IDs for cross-session testing
export SESSION_1="session-alpha-$(date +%s)"
export SESSION_2="session-beta-$(date +%s)"
export SESSION_3="session-gamma-$(date +%s)"

echo "Session 1: $SESSION_1"
echo "Session 2: $SESSION_2"
echo "Session 3: $SESSION_3"
```

### 2. Import Required Modules

```python
import sys
sys.path.insert(0, 'resources/scripts')

from session_memory import TripleLayerMemory
import sqlite3
import json
import time
import os
```

---

## Test Cases

### Test Case 1: Long-Term Persistence

**Objective**: Verify memories persist across multiple sessions and database connections.

```python
def test_long_term_persistence():
    """Test long-term memory persistence"""
    db_path = '.agentdb/test-longterm.db'
    session_id = os.environ['SESSION_1']

    # Phase 1: Store long-term memories
    print("Phase 1: Storing long-term memories...")
    memory = TripleLayerMemory(db_path)

    core_facts = [
        ("User's name is Alice", 0.95, {'type': 'user_info'}),
        ("Preferred language: Python", 0.90, {'type': 'preference'}),
        ("Expert in machine learning", 0.85, {'type': 'expertise'}),
        ("Works in data science", 0.88, {'type': 'profession'}),
        ("Timezone: UTC-5", 0.92, {'type': 'settings'}),
    ]

    stored_ids = []
    for content, priority, metadata in core_facts:
        entry_id = memory.store(content, session_id, priority, metadata=metadata)
        stored_ids.append(entry_id)
        print(f"  ✅ Stored: {content}")

    memory.close()
    print(f"Stored {len(stored_ids)} facts")

    # Phase 2: Close and reopen database
    print("\nPhase 2: Reopening database...")
    time.sleep(1)  # Brief pause

    memory = TripleLayerMemory(db_path)

    # Retrieve memories
    retrieved = memory.retrieve(session_id, layer='long_term', limit=10)
    print(f"Retrieved {len(retrieved)} long-term memories")

    # Verify all facts present
    retrieved_contents = {m['content'] for m in retrieved}
    expected_contents = {fact[0] for fact in core_facts}

    print("\nVerification:")
    all_found = True
    for content in expected_contents:
        found = content in retrieved_contents
        print(f"  {'✅' if found else '❌'} {content}")
        all_found = all_found and found

    memory.close()

    print(f"\nTest Case 1: {'PASSED' if all_found else 'FAILED'}")
    return all_found
```

**Expected Results**:
- All 5 core facts stored successfully
- All facts retrieved after database reopen
- Long-term layer assignment verified
- Data integrity maintained

---

### Test Case 2: Cross-Session Context

**Objective**: Test memory retrieval across different sessions.

```python
def test_cross_session_context():
    """Test cross-session memory access"""
    db_path = '.agentdb/test-longterm.db'
    session_1 = os.environ['SESSION_1']
    session_2 = os.environ['SESSION_2']
    session_3 = os.environ['SESSION_3']

    memory = TripleLayerMemory(db_path)

    # Store memories in different sessions
    print("Storing memories across sessions...")

    # Session 1: User preferences
    memory.store("Prefers dark mode", session_1, 0.85,
                 metadata={'session': 1, 'type': 'preference'})
    memory.store("Uses VS Code", session_1, 0.80,
                 metadata={'session': 1, 'type': 'tool'})

    # Session 2: Project context
    memory.store("Working on ML project", session_2, 0.75,
                 metadata={'session': 2, 'type': 'context'})
    memory.store("Uses TensorFlow", session_2, 0.70,
                 metadata={'session': 2, 'type': 'framework'})

    # Session 3: Learning topics
    memory.store("Learning about transformers", session_3, 0.65,
                 metadata={'session': 3, 'type': 'learning'})

    print("\nRetrieving memories by session:")

    # Test retrieval for each session
    results = {}
    for i, session in enumerate([session_1, session_2, session_3], 1):
        memories = memory.retrieve(session, limit=10)
        results[f'session_{i}'] = len(memories)
        print(f"  Session {i}: {len(memories)} memories")

    # Verify isolation
    session_1_memories = memory.retrieve(session_1, limit=10)
    session_1_contents = [m['content'] for m in session_1_memories]

    isolated = ("Working on ML project" not in session_1_contents and
                "Learning about transformers" not in session_1_contents)

    print(f"\n{'✅' if isolated else '❌'} Session isolation maintained")

    memory.close()

    passed = (results['session_1'] >= 2 and
              results['session_2'] >= 2 and
              results['session_3'] >= 1 and
              isolated)

    print(f"\nTest Case 2: {'PASSED' if passed else 'FAILED'}")
    return passed
```

**Expected Results**:
- Session 1: 2+ memories
- Session 2: 2+ memories
- Session 3: 1+ memory
- Sessions properly isolated
- No cross-contamination

---

### Test Case 3: Data Integrity and Corruption Recovery

**Objective**: Test database integrity and recovery mechanisms.

```python
def test_data_integrity():
    """Test data integrity and corruption recovery"""
    db_path = '.agentdb/test-longterm.db'
    session_id = os.environ['SESSION_1']

    memory = TripleLayerMemory(db_path)

    # Store test data
    print("Storing test data...")
    test_content = "Integrity test entry"
    entry_id = memory.store(test_content, session_id, 0.9)

    # Verify storage
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, content, priority, layer
        FROM memory_layers
        WHERE id = ?
    ''', (entry_id,))

    row = cursor.fetchone()
    print(f"Stored entry: ID={row[0]}, Content={row[1]}, "
          f"Priority={row[2]}, Layer={row[3]}")

    # Test 1: Verify indexes exist
    cursor.execute('''
        SELECT name FROM sqlite_master
        WHERE type='index' AND tbl_name='memory_layers'
    ''')
    indexes = [row[0] for row in cursor.fetchall()]

    expected_indexes = ['idx_layer_session', 'idx_expires', 'idx_priority']
    indexes_ok = all(idx in indexes for idx in expected_indexes)
    print(f"\n{'✅' if indexes_ok else '❌'} Required indexes present")

    # Test 2: Verify foreign key integrity
    cursor.execute('PRAGMA foreign_keys')
    fk_enabled = cursor.fetchone()[0] == 1
    print(f"{'✅' if fk_enabled else '❌'} Foreign keys enabled: {fk_enabled}")

    # Test 3: Check for orphaned records
    cursor.execute('''
        SELECT COUNT(*) FROM memory_layers
        WHERE expires_at < ?
    ''', (int(time.time()),))
    expired = cursor.fetchone()[0]
    print(f"Expired entries: {expired}")

    # Test 4: Integrity check
    cursor.execute('PRAGMA integrity_check')
    integrity = cursor.fetchone()[0]
    integrity_ok = integrity == 'ok'
    print(f"{'✅' if integrity_ok else '❌'} Integrity check: {integrity}")

    conn.close()
    memory.close()

    passed = indexes_ok and integrity_ok

    print(f"\nTest Case 3: {'PASSED' if passed else 'FAILED'}")
    return passed
```

**Expected Results**:
- All required indexes present
- Database integrity check passes
- No corruption detected
- Foreign keys properly configured

---

### Test Case 4: Backup and Recovery

**Objective**: Test backup creation and restoration.

```python
def test_backup_recovery():
    """Test backup and recovery mechanisms"""
    db_path = '.agentdb/test-longterm.db'
    backup_path = '.agentdb/test-longterm-backup.db'
    session_id = os.environ['SESSION_1']

    memory = TripleLayerMemory(db_path)

    # Store original data
    print("Storing original data...")
    original_data = [
        ("Backup test 1", 0.9),
        ("Backup test 2", 0.8),
        ("Backup test 3", 0.7),
    ]

    for content, priority in original_data:
        memory.store(content, session_id, priority)

    original_count = len(memory.retrieve(session_id, limit=100))
    print(f"Original entries: {original_count}")

    memory.close()

    # Create backup
    print("\nCreating backup...")
    import shutil
    shutil.copy2(db_path, backup_path)
    print(f"Backup created: {backup_path}")

    # Modify original database
    print("\nModifying original database...")
    memory = TripleLayerMemory(db_path)
    memory.store("New entry after backup", session_id, 0.6)
    modified_count = len(memory.retrieve(session_id, limit=100))
    print(f"Modified entries: {modified_count}")
    memory.close()

    # Restore from backup
    print("\nRestoring from backup...")
    shutil.copy2(backup_path, db_path)

    # Verify restoration
    memory = TripleLayerMemory(db_path)
    restored = memory.retrieve(session_id, limit=100)
    restored_count = len(restored)

    restored_contents = {m['content'] for m in restored}
    new_entry_absent = "New entry after backup" not in restored_contents

    print(f"Restored entries: {restored_count}")
    print(f"{'✅' if new_entry_absent else '❌'} Post-backup entry correctly absent")

    # Cleanup
    os.remove(backup_path)
    memory.close()

    passed = (restored_count == original_count and new_entry_absent)

    print(f"\nTest Case 4: {'PASSED' if passed else 'FAILED'}")
    return passed
```

**Expected Results**:
- Backup created successfully
- Original data preserved in backup
- Post-backup modifications not in restored version
- Restoration completes without errors

---

### Test Case 5: Memory Export and Import

**Objective**: Test JSON export/import functionality.

```python
def test_export_import():
    """Test memory export and import"""
    db_path = '.agentdb/test-longterm.db'
    export_file = '.agentdb/export-test.json'
    session_id = os.environ['SESSION_1']

    memory = TripleLayerMemory(db_path)

    # Export memories
    print("Exporting memories to JSON...")

    memories = memory.retrieve(session_id, limit=100)
    export_data = []

    for mem in memories:
        export_data.append({
            'layer': mem['layer'],
            'content': mem['content'],
            'priority': mem['priority'],
            'metadata': mem['metadata']
        })

    with open(export_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    export_count = len(export_data)
    print(f"Exported {export_count} memories to {export_file}")

    # Clear some memories (for import test)
    conn = sqlite3.connect(db_path)
    conn.execute('DELETE FROM memory_layers WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()

    print(f"\nCleared original memories")

    # Import memories
    print("Importing memories from JSON...")

    with open(export_file, 'r') as f:
        import_data = json.load(f)

    new_session = "imported-session"
    for mem in import_data:
        memory.store(
            mem['content'],
            new_session,
            mem['priority'],
            metadata=mem.get('metadata')
        )

    imported = memory.retrieve(new_session, limit=100)
    import_count = len(imported)

    print(f"Imported {import_count} memories")

    # Verify content matches
    exported_contents = {m['content'] for m in export_data}
    imported_contents = {m['content'] for m in imported}

    content_match = exported_contents == imported_contents
    print(f"{'✅' if content_match else '❌'} Content matches after import")

    # Cleanup
    os.remove(export_file)
    memory.close()

    passed = (import_count == export_count and content_match)

    print(f"\nTest Case 5: {'PASSED' if passed else 'FAILED'}")
    return passed
```

**Expected Results**:
- Export creates valid JSON
- All memories exported
- Import restores all memories
- Content matches after round-trip

---

## Running the Test Suite

```python
def run_longterm_storage_tests():
    """Run all long-term storage tests"""
    print("=" * 60)
    print("AgentDB Long-Term Storage Test Suite")
    print("=" * 60)

    tests = [
        ("Long-Term Persistence", test_long_term_persistence),
        ("Cross-Session Context", test_cross_session_context),
        ("Data Integrity", test_data_integrity),
        ("Backup & Recovery", test_backup_recovery),
        ("Export & Import", test_export_import),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print('=' * 60)

        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)

    print(f"\nOverall: {passed_count}/{total} tests passed")
    print(f"Success Rate: {(passed_count/total)*100:.1f}%")

    return all(p for _, p in results)

# Execute
if __name__ == '__main__':
    success = run_longterm_storage_tests()
    sys.exit(0 if success else 1)
```

---

## Cleanup

```bash
# Remove test artifacts
rm -rf .agentdb/test-longterm.db
rm -rf .agentdb/test-longterm-backup.db
rm -rf .agentdb/export-test.json

# Unset environment variables
unset TEST_DB SESSION_1 SESSION_2 SESSION_3
```

---

## Success Criteria

- ✅ All 5 test cases pass
- ✅ Long-term persistence verified across sessions
- ✅ Cross-session isolation maintained
- ✅ Data integrity checks pass
- ✅ Backup/restore works correctly
- ✅ Export/import preserves data

---

## Performance Benchmarks

Expected performance characteristics:

- **Storage**: < 5ms per entry
- **Retrieval**: < 10ms for 100 entries
- **Backup**: < 100ms for 1000 entries
- **Export**: < 50ms for 100 entries
- **Import**: < 100ms for 100 entries

---

## Notes

- Database uses WAL mode for better concurrency
- Indexes optimize common query patterns
- Backup is atomic (no partial states)
- Export format is JSON for portability
- Cross-session access requires explicit session ID
